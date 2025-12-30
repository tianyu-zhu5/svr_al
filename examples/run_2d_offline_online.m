% 2D offline + online refinement demo (M5).

thisFileDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(thisFileDir, "..");
addpath(fullfile(projectRoot, "src"));

cfg = struct();
cfg.seed = 7;
cfg.Npool = 30000;
cfg.N0 = 20;
cfg.budgetTotal = 100;
cfg.bootstrapM = 15;
cfg.weights = struct("tauQuantile", 0.30, "form", "inv");
cfg.stop = struct("useUminStable", true, "UminDelta", 1e-3, "K", 3, "usePfConv", true, "PfEta", 1e-2, "minIters", 12);

% Online (M5) settings
cfg.online = struct();
cfg.online.deltaX = 0.05;     % trigger if design point changes less than this (physical space)
cfg.online.Utrigger = 1.8;    % trigger if Umin in neighborhood is small
cfg.online.alpha = 0.93;      % strong exploitation online
cfg.online.r0 = 0.12;         % initial neighborhood radius in [0,1]^d
cfg.online.rMin = 0.03;
cfg.online.rShrink = 0.85;
cfg.online.maxIters = 12;

rng(cfg.seed, "twister");

spec = struct();
spec.d = 2;
spec.lb = [-3, -3];
spec.ub = [ 3,  3];
spec.dist(1) = struct("type","normal", "params", struct("mu",0, "sigma",1));
spec.dist(2) = struct("type","normal", "params", struct("mu",0, "sigma",1));

spec.trueG = @(X) 1.5 - (X(:,1).^2 + X(:,2).^2);
spec.evalFcn = spec.trueG;

outDir = lqy.io.makeOutDir(fullfile(projectRoot, "out"), "2d_offline_online");

% ----- Offline stage (reuse logic from run_2d_offline.m) -----
pool = lqy.pool.buildPoolTruncatedLHS(spec, cfg);
normModelTmp = struct("lb", spec.lb, "ub", spec.ub);
[X0, idx0] = lqy.doe.selectDOE_kmeans(pool, cfg, normModelTmp);
g0 = spec.evalFcn(X0);

log = struct();
log.X_all = X0;
log.g_all = g0;
log.stage = zeros(size(X0,1), 1);
log.meta = struct("seed", cfg.seed, "cfg", cfg, "spec", rmfield(spec, ["evalFcn","trueG"]));

XTrain = X0;
gTrain = g0;
poolX = pool.X;
picked = false(size(poolX,1), 1);
picked(idx0) = true;

Pf_hat = [];
UminHist = [];
alphaHist = [];
NevalHist = [];

while true
  normModel = lqy.norm.fitNormModel(XTrain, spec);
  model = lqy.svr.trainBootstrapSurrogate(XTrain, gTrain, cfg, normModel);
  [mu, sigma] = lqy.svr.predictBootstrap(model, poolX);

  Pf_hat(end+1,1) = lqy.reliability.estimatePf(mu, cfg); %#ok<SAGROW>
  U = abs(mu) ./ max(sigma, 1e-12);
  UminHist(end+1,1) = min(U); %#ok<SAGROW>

  t = numel(Pf_hat);
  [score, detail] = lqy.al.acquisitionScore(mu, sigma, poolX, XTrain, cfg, normModel, t);
  alphaHist(end+1,1) = detail.alpha; %#ok<SAGROW>

  score(picked) = -inf;
  [~, nextIdx] = max(score);
  xNext = poolX(nextIdx, :);
  gNext = spec.evalFcn(xNext);

  XTrain = [XTrain; xNext]; %#ok<AGROW>
  gTrain = [gTrain; gNext]; %#ok<AGROW>
  picked(nextIdx) = true;
  log.stage = [log.stage; 1]; %#ok<AGROW>

  Neval = size(XTrain, 1);
  NevalHist(end+1,1) = Neval; %#ok<SAGROW>
  state = struct("Neval", Neval, "PfHist", Pf_hat, "UminHist", UminHist);
  [stop, reason] = lqy.al.stopCriteria(state, cfg);
  if stop
    fprintf("Offline stop: %s (Neval=%d)\\n", reason, Neval);
    break;
  end
end

log.X_all = XTrain;
log.g_all = gTrain;
log.Pf_hat = Pf_hat;
log.Umin = UminHist;
log.alpha = alphaHist;
log.NevalHist = NevalHist;

% ----- Online stage (simulate a design point trajectory) -----
% In real use, designHist comes from RBDO iterations.
designHist = [-0.4 -0.2; -0.2 -0.1; -0.1 -0.05; -0.08 -0.04; -0.075 -0.038];

budgetRemain = cfg.budgetTotal - size(log.X_all, 1);
[model, log, used, info] = lqy.online.onlineRefine(spec, pool, model, log, designHist, budgetRemain, cfg);
fprintf("Online refine triggered=%d reason=%s usedBudget=%d iters=%d\\n", info.triggered, info.triggerReason, used, info.iters);

stageToSave = "offline";
if used > 0
  stageToSave = "online";
end

lqy.io.saveModelAndLog(model, log, outDir, stageToSave);
lqy.viz.plot2DResults(spec, pool, log, outDir);

disp("Offline+Online demo complete.");
disp("Output: " + outDir);

