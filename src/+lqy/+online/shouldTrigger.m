function [trigger, reason, detail] = shouldTrigger(designHist, model, poolX, XTrain, cfg)
%SHOULDTRIGGER Decide whether to enter online refinement near design point.

arguments
  designHist (:,:) double
  model (1,1) struct
  poolX (:,:) double
  XTrain (:,:) double
  cfg (1,1) struct
end

trigger = false;
reason = "none";
detail = struct();

if isempty(designHist)
  return;
end

normModel = model.normModel;
xd = designHist(end, :);

deltaX = 1e-2;
if isfield(cfg, "online") && isfield(cfg.online, "deltaX")
  deltaX = cfg.online.deltaX;
end

if size(designHist, 1) >= 2
  prev = designHist(end-1, :);
  d = norm(xd - prev);
  detail.deltaX = d;
  if d < deltaX
    trigger = true;
    reason = "design_stable";
    return;
  end
end

Utrigger = 2.0;
if isfield(cfg, "online") && isfield(cfg.online, "Utrigger")
  Utrigger = cfg.online.Utrigger;
end

r = 0.10;
if isfield(cfg, "online") && isfield(cfg.online, "r0")
  r = cfg.online.r0;
end

idxLocal = lqy.online.localPoolIndices(poolX, xd, r, normModel);
detail.localCount = numel(idxLocal);

if isempty(idxLocal)
  return;
end

[ghat, sigma] = lqy.surrogate.predictSurrogate(model, poolX(idxLocal, :));
U = abs(ghat) ./ max(sigma, 1e-12);
UminLocal = min(U);
detail.UminLocal = UminLocal;

if UminLocal < Utrigger
  trigger = true;
  reason = "U_low_near_design";
end
end
