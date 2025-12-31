function main = trainMainModel(ZTrain, yTrain, w, cfg, hp)
%TRAINMAINMODEL Train a single, controllable surrogate model for g-hat.
%
% Uses SVR if configured and available; otherwise uses (weighted) KRR.

arguments
  ZTrain (:,:) double
  yTrain (:,1) double
  w (:,1) double
  cfg (1,1) struct
  hp (1,1) struct
end

method = inferMethod(cfg, hp);

if method == "svr"
  if exist("fitrsvm", "file") ~= 2
    error("cfg.surrogate.method='svr' requires fitrsvm.");
  end
  mustHave(hp, ["KernelScale","BoxConstraint","Epsilon"]);
  main = fitrsvm(ZTrain, yTrain, ...
    "KernelFunction", "gaussian", ...
    "KernelScale", hp.KernelScale, ...
    "BoxConstraint", hp.BoxConstraint, ...
    "Epsilon", hp.Epsilon, ...
    "Standardize", false, ...
    "Weights", w);
  return;
end

% KRR (toolbox-free)
if ~isfield(hp, "ell") || ~isfield(hp, "lambda")
  error("KRR hp requires hp.ell and hp.lambda.");
end
main = lqy.svr.trainKRRWeighted(ZTrain, yTrain, struct("ell",hp.ell,"lambda",hp.lambda), w);
end

function method = inferMethod(cfg, hp)
% Prefer hp.type if provided; fall back to cfg and availability.
if isfield(hp, "type")
  t = lower(string(hp.type));
  if t == "svr"
    method = "svr";
    return;
  end
  if t == "krr"
    method = "krr";
    return;
  end
end

method = "auto";
if isfield(cfg, "surrogate") && isfield(cfg.surrogate, "method")
  method = string(cfg.surrogate.method);
end
method = lower(method);
switch method
  case "svr"
    method = "svr";
  case "krr"
    method = "krr";
  otherwise
    if exist("fitrsvm", "file") == 2
      method = "svr";
    else
      method = "krr";
    end
end
end

function mustHave(s, names)
for k = 1:numel(names)
  if ~isfield(s, names(k))
    error("Missing required hyperparam: %s", names(k));
  end
end
end
