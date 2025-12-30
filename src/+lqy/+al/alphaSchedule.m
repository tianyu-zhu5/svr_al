function alpha = alphaSchedule(t, cfg)
%ALPHASCHEDULE Simple alpha schedule from exploration to exploitation.

arguments
  t (1,1) double {mustBeInteger, mustBeNonnegative}
  cfg (1,1) struct
end

a0 = 0.3;
a1 = 0.8;
T = 30;

if isfield(cfg, "acq") && isfield(cfg.acq, "alpha0"); a0 = cfg.acq.alpha0; end
if isfield(cfg, "acq") && isfield(cfg.acq, "alpha1"); a1 = cfg.acq.alpha1; end
if isfield(cfg, "acq") && isfield(cfg.acq, "alphaRampIters"); T = cfg.acq.alphaRampIters; end

alpha = a0 + (a1 - a0) .* min(t ./ max(T,1), 1);
alpha = min(max(alpha, 0), 1);
end

