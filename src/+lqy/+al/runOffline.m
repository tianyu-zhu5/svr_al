function [model, log, pool] = runOffline(spec, cfg, outDir)
%RUNOFFLINE Offline active learning loop with checkpointing and resumability.
%
% Resumes if cfg.resume==true and outDir contains pool.mat + log.mat.

arguments
  spec (1,1) struct
  cfg (1,1) struct
  outDir (1,1) string
end

if ~exist(outDir, "dir")
  mkdir(outDir);
end

resume = false;
if isfield(cfg, "resume"); resume = logical(cfg.resume); end

poolPath = fullfile(outDir, "pool.mat");
logPath = fullfile(outDir, "log.mat");

if resume && exist(poolPath, "file") == 2 && exist(logPath, "file") == 2
  S = load(poolPath, "pool");
  pool = S.pool;
  S = load(logPath, "log");
  log = S.log;
else
  pool = lqy.pool.buildPoolTruncatedLHS(spec, cfg);
  save(poolPath, "pool");

  normModelTmp = struct("lb", spec.lb, "ub", spec.ub);
  [~, idx0] = lqy.doe.selectDOE_kmeans(pool, cfg, normModelTmp);
  X0 = pool.X(idx0, :);
  g0 = spec.evalFcn(X0);

  log = struct();
  log.poolIdx_all = idx0(:);
  log.X_all = X0;
  log.g_all = g0(:);
  log.stage = zeros(numel(idx0), 1);
  log.Pf_hat = [];
  log.Umin = [];
  log.alpha = [];
  log.NevalHist = [];
  log.theta = {};
  log.tune = struct();
  log.meta = struct("seed", getfieldWithDefault(cfg, "seed", NaN), "cfg", cfg, "spec", rmfieldOptional(spec, ["evalFcn","trueG"]));
  save(logPath, "log");
end

poolX = pool.X;
picked = false(size(poolX,1), 1);
picked(log.poolIdx_all) = true;

XTrain = log.X_all;
gTrain = log.g_all;
hpCurrent = [];
if isfield(log, "tune") && isfield(log.tune, "hpLast") && ~isempty(log.tune.hpLast)
  hpCurrent = log.tune.hpLast;
end

while true
  normModel = lqy.norm.fitNormModel(XTrain, spec);
  w = lqy.svr.computeWeights(gTrain, cfg);

  [doTune, hpCurrent, tuneInfo] = maybeTune(normModel, XTrain, gTrain, w, cfg, hpCurrent);
  model = lqy.surrogate.trainSurrogate(XTrain, gTrain, cfg, spec, normModel, hpCurrent);

  [ghat, sigma] = lqy.surrogate.predictSurrogate(model, poolX);
  Pf = lqy.reliability.estimatePf(ghat, cfg);
  U = abs(ghat) ./ max(sigma, 1e-12);
  Umin = min(U);

  t = numel(log.Pf_hat) + 1;
  [score, detail] = lqy.al.acquisitionScore(ghat, sigma, poolX, XTrain, cfg, normModel, t);
  score(picked) = -inf;
  [~, nextIdx] = max(score);

  xNext = poolX(nextIdx, :);
  gNext = spec.evalFcn(xNext);

  % Append
  picked(nextIdx) = true;
  log.poolIdx_all(end+1,1) = nextIdx; %#ok<AGROW>
  log.X_all(end+1,:) = xNext; %#ok<AGROW>
  log.g_all(end+1,1) = gNext; %#ok<AGROW>
  log.stage(end+1,1) = 1; %#ok<AGROW>

  log.Pf_hat(end+1,1) = Pf; %#ok<AGROW>
  log.Umin(end+1,1) = Umin; %#ok<AGROW>
  log.alpha(end+1,1) = detail.alpha; %#ok<AGROW>
  log.NevalHist(end+1,1) = size(log.X_all,1); %#ok<AGROW>
  if doTune
    log.theta{end+1,1} = hpCurrent; %#ok<AGROW>
  else
    log.theta{end+1,1} = struct("type","reuse"); %#ok<AGROW>
  end
  log.tune.hpLast = hpCurrent;
  log.tune.lastInfo = tuneInfo;

  % Checkpoint
  save(logPath, "log");
  lqy.io.saveModelAndLog(model, log, outDir, "offline");

  % Stop?
  state = struct("Neval", size(log.X_all,1), "PfHist", log.Pf_hat, "UminHist", log.Umin);
  [stop, ~] = lqy.al.stopCriteria(state, cfg);
  if stop
    break;
  end

  XTrain = log.X_all;
  gTrain = log.g_all;
end
end

function [doTune, hp, info] = maybeTune(normModel, XTrain, gTrain, w, cfg, hpPrev)
doTune = false;
info = struct("method","none");
hp = hpPrev;

if ~isfield(cfg, "tune") || ~isfield(cfg.tune, "enabled") || ~cfg.tune.enabled
  if isempty(hp)
    hp = defaultHpForMain(normModel, XTrain, gTrain, cfg);
  end
  return;
end

minN = 20;
everyNeval = 10;
methodHint = "auto";
if isfield(cfg.tune, "minTrainN"); minN = cfg.tune.minTrainN; end
if isfield(cfg.tune, "everyNeval"); everyNeval = cfg.tune.everyNeval; end
if isfield(cfg.tune, "methodHint"); methodHint = string(cfg.tune.methodHint); end

Neval = size(XTrain,1);
if Neval < minN
  return;
end

if isempty(hpPrev)
  if ~(Neval == minN || mod(Neval, everyNeval) == 0)
    return;
  end
else
  if mod(Neval, everyNeval) ~= 0
    return;
  end
end

Z = lqy.norm.zscoreApply(XTrain, normModel);
[hp, info] = lqy.svr.tuneHyperparams(Z, gTrain, w, cfg, methodHint);
doTune = true;
end

function hp = defaultHpForMain(normModel, XTrain, gTrain, cfg)
Z = lqy.norm.zscoreApply(XTrain, normModel);
base = lqy.svr.defaultHyperparams(Z, gTrain, cfg);

method = "auto";
if isfield(cfg, "surrogate") && isfield(cfg.surrogate, "method")
  method = string(cfg.surrogate.method);
end
method = lower(method);

useSVR = (method == "svr") || (method == "auto" && exist("fitrsvm", "file") == 2);
if method == "krr"; useSVR = false; end

if useSVR
  eps0 = 0.1 * std(gTrain, 0, 1, "omitnan");
  if ~isfinite(eps0) || eps0 <= 0; eps0 = 1e-3; end
  hp = struct("type","svr", "KernelScale", base.ell, "BoxConstraint", 10, "Epsilon", eps0);
else
  hp = struct("type","krr", "ell", base.ell, "lambda", base.lambda);
end
end

function out = getfieldWithDefault(s, name, defaultValue)
if isfield(s, name)
  out = s.(name);
else
  out = defaultValue;
end
end

function s2 = rmfieldOptional(s, names)
s2 = s;
for i = 1:numel(names)
  if isfield(s2, names(i))
    s2 = rmfield(s2, names(i));
  end
end
end
