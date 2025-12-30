function [model, log, usedBudget, info] = onlineRefine(spec, pool, model, log, designHist, budgetRemain, cfg)
%ONLINEREFINE Perform local refinement near current design point.

arguments
  spec (1,1) struct
  pool (1,1) struct
  model (1,1) struct
  log (1,1) struct
  designHist (:,:) double
  budgetRemain (1,1) double {mustBeInteger, mustBeNonnegative}
  cfg (1,1) struct
end

info = struct("triggered", false, "triggerReason", "none", "iters", 0);
usedBudget = 0;

if budgetRemain <= 0 || isempty(designHist)
  return;
end

poolX = pool.X;

if ~isfield(log, "poolIdx_all") || isempty(log.poolIdx_all)
  error("log.poolIdx_all is required for reliable online refinement.");
end

picked = false(size(poolX,1), 1);
picked(log.poolIdx_all) = true;

XTrain = log.X_all;
gTrain = log.g_all;

[trigger, reason] = lqy.online.shouldTrigger(designHist, model, poolX, XTrain, cfg);
if ~trigger
  info.triggered = false;
  info.triggerReason = reason;
  return;
end

info.triggered = true;
info.triggerReason = reason;

alpha = 0.92;
if isfield(cfg, "online") && isfield(cfg.online, "alpha")
  alpha = cfg.online.alpha;
end

r = 0.10;
rMin = 0.03;
rShrink = 0.8;
maxIters = budgetRemain;

if isfield(cfg, "online") && isfield(cfg.online, "r0"); r = cfg.online.r0; end
if isfield(cfg, "online") && isfield(cfg.online, "rMin"); rMin = cfg.online.rMin; end
if isfield(cfg, "online") && isfield(cfg.online, "rShrink"); rShrink = cfg.online.rShrink; end
if isfield(cfg, "online") && isfield(cfg.online, "maxIters"); maxIters = min(maxIters, cfg.online.maxIters); end

xd = designHist(end, :);

hpCurrent = [];
if isfield(model, "hpShared")
  hpCurrent = model.hpShared;
end
onlineTune = false;
if isfield(cfg, "online") && isfield(cfg.online, "tuneEnabled")
  onlineTune = logical(cfg.online.tuneEnabled);
end

for it = 1:maxIters
  % Update normalization and surrogate with newest samples.
  normModel = lqy.norm.fitNormModel(XTrain, spec);
  if onlineTune && isfield(cfg, "tune") && isfield(cfg.tune, "enabled") && cfg.tune.enabled
    w = lqy.svr.computeWeights(gTrain, cfg);
    Z = lqy.norm.zscoreApply(XTrain, normModel);
    [hpCurrent, ~] = lqy.svr.tuneHyperparams(Z, gTrain, w, cfg, "auto");
  end
  model = lqy.svr.trainBootstrapSurrogate(XTrain, gTrain, cfg, normModel, hpCurrent);

  idxLocal = lqy.online.localPoolIndices(poolX, xd, r, normModel);
  if isempty(idxLocal)
    if r <= rMin
      break;
    end
    r = max(r * rShrink, rMin);
    continue;
  end

  Xcand = poolX(idxLocal, :);
  [mu, sigma] = lqy.svr.predictBootstrap(model, Xcand);

  localCfg = cfg;
  if ~isfield(localCfg, "acq"); localCfg.acq = struct(); end
  localCfg.acq.alpha0 = alpha;
  localCfg.acq.alpha1 = alpha;
  localCfg.acq.alphaRampIters = 1;

  t = it;
  [score, ~] = lqy.al.acquisitionScore(mu, sigma, Xcand, XTrain, localCfg, normModel, t);

  % Exclude already selected pool points.
  score(picked(idxLocal)) = -inf;
  [bestScore, relIdx] = max(score);
  if ~isfinite(bestScore)
    if r <= rMin
      break;
    end
    r = max(r * rShrink, rMin);
    continue;
  end

  nextIdx = idxLocal(relIdx);
  xNext = poolX(nextIdx, :);
  gNext = spec.evalFcn(xNext);

  XTrain = [XTrain; xNext]; %#ok<AGROW>
  gTrain = [gTrain; gNext]; %#ok<AGROW>
  picked(nextIdx) = true;

  log.poolIdx_all(end+1,1) = nextIdx; %#ok<AGROW>
  log.X_all = XTrain;
  log.g_all = gTrain;
  log.stage = [log.stage; 2]; %#ok<AGROW>

  usedBudget = usedBudget + 1;
  info.iters = it;
  if usedBudget >= budgetRemain
    break;
  end

  % Shrink neighborhood slowly as we add points.
  r = max(r * rShrink, rMin);
end
end
