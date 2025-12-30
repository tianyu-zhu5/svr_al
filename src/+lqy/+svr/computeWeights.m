function w = computeWeights(gTrain, cfg)
%COMPUTEWEIGHTS Compute sampling weights (larger near |g| ~ 0).

arguments
  gTrain (:,1) double
  cfg (1,1) struct
end

absG = abs(gTrain);

tau = [];
if isfield(cfg, "weights") && isfield(cfg.weights, "tau")
  tau = cfg.weights.tau;
end

if isempty(tau)
  q = 0.30;
  if isfield(cfg, "weights") && isfield(cfg.weights, "tauQuantile")
    q = cfg.weights.tauQuantile;
  end
  if exist("quantile", "file") == 2
    tau = quantile(absG, q);
  else
    tau = lqy.util.quantile1(absG, q);
  end
  if ~isfinite(tau) || tau <= 0
    tau = max(median(absG, "omitnan"), 1);
  end
end

form = "inv";
if isfield(cfg, "weights") && isfield(cfg.weights, "form")
  form = string(cfg.weights.form);
end

switch lower(form)
  case "inv"
    % w = exp(-|g|/tau)
    w = exp(-absG ./ max(tau, 1e-12));
  case "lin"
    % w = 1/(|g|+tau)
    w = 1 ./ (absG + max(tau, 1e-12));
  otherwise
    error("Unsupported weights.form: %s", form);
end

w(~isfinite(w)) = 0;
w = max(w, 0);
if all(w == 0)
  w(:) = 1;
end
end
