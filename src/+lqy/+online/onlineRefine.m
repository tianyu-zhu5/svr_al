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

XTrain = log.X_all;
gTrain = log.g_all;
poolX = pool.X;

% Track already-picked pool points by exact match on rows (pool-based assumption).
picked = false(size(poolX,1), 1);
% Best-effort mapping: for each training point, find identical pool row.
% (For robustness, user should store pool indices; we keep this lightweight for now.)
for i = 1:size(XTrain,1)
  hit = find(all(poolX == XTrain(i,:), 2), 1, "first");
  if ~isempty(hit); picked(hit) = true; end
end

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

for it = 1:maxIters
  % Update normalization and surrogate with newest samples.
  normModel = lqy.norm.fitNormModel(XTrain, spec);
  model = lqy.svr.trainBootstrapSurrogate(XTrain, gTrain, cfg, normModel);

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

  % Exclude already selected points.
  pickedLocal = false(size(Xcand,1), 1);
  for k = 1:numel(idxLocal)
    if picked(idxLocal(k))
      pickedLocal(k) = true;
    end
  end
  score(pickedLocal) = -inf;
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

