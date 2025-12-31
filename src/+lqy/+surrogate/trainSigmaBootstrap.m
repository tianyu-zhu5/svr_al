function ens = trainSigmaBootstrap(ZTrain, yTrain, w, cfg, hp)
%TRAINSIGMABOOTSTRAP Train bootstrap ensemble used ONLY for sigma(x).
%
% Main prediction should come from a separate "main" model.

arguments
  ZTrain (:,:) double
  yTrain (:,1) double
  w (:,1) double
  cfg (1,1) struct
  hp (1,1) struct
end

M = 20;
if isfield(cfg, "bootstrapM"); M = cfg.bootstrapM; end

N = size(ZTrain, 1);
p = w(:);
p(~isfinite(p) | p < 0) = 0;
if all(p == 0); p(:) = 1; end
p = p ./ sum(p);

useSVR = false;
if isfield(hp, "type") && lower(string(hp.type)) == "svr"
  useSVR = true;
elseif isfield(hp, "KernelScale")
  useSVR = true;
else
  method = "auto";
  if isfield(cfg, "surrogate") && isfield(cfg.surrogate, "method")
    method = string(cfg.surrogate.method);
  end
  method = lower(method);
  useSVR = (method == "svr") || (method == "auto" && exist("fitrsvm", "file") == 2);
  if method == "krr"; useSVR = false; end
end

models = cell(M, 1);
for m = 1:M
  idx = lqy.util.weightedResample(p, N);
  Zb = ZTrain(idx,:);
  yb = yTrain(idx);

  if useSVR
    if exist("fitrsvm", "file") ~= 2
      error("Bootstrap sigma SVR requested but fitrsvm is unavailable.");
    end
    mdl = fitrsvm(Zb, yb, ...
      "KernelFunction", "gaussian", ...
      "KernelScale", hp.KernelScale, ...
      "BoxConstraint", hp.BoxConstraint, ...
      "Epsilon", hp.Epsilon, ...
      "Standardize", false);
    models{m} = mdl;
  else
    base = lqy.svr.trainKRR(Zb, yb, struct("ell",hp.ell,"lambda",hp.lambda));
    models{m} = base;
  end
end

ens = struct();
ens.models = models;
ens.meta = struct("bootstrapM", M, "useSVR", useSVR);
end
