function Pf = estimatePf(ghat, cfg)
%ESTIMATEPF Estimate failure probability from surrogate prediction on a fixed sample set.
% Default (A-mean): Pf = mean(ghat <= 0).

arguments
  ghat (:,1) double
  cfg (1,1) struct
end

useSafety = false;
kappa = 0;
if isfield(cfg, "reliability") && isfield(cfg.reliability, "useSafety")
  useSafety = logical(cfg.reliability.useSafety);
end
if isfield(cfg, "reliability") && isfield(cfg.reliability, "kappa")
  kappa = cfg.reliability.kappa;
end

if useSafety
  error("Safety-side Pf requires sigma; call estimatePfSafety(mu,sigma,...) or disable useSafety.");
end

Pf = mean(ghat <= 0);
end
