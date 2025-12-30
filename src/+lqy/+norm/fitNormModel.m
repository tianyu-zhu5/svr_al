function normModel = fitNormModel(XTrain, spec)
%FITNORMMODEL Fit [0,1]^d scaling (from spec) and z-score (from training set).

arguments
  XTrain (:,:) double
  spec (1,1) struct
end

if ~isfield(spec, "lb") || ~isfield(spec, "ub")
  error("spec.lb and spec.ub are required.");
end

lb = spec.lb(:).';
ub = spec.ub(:).';

if size(XTrain,2) ~= numel(lb)
  error("XTrain columns (%d) must match numel(spec.lb) (%d).", size(XTrain,2), numel(lb));
end

sigma01 = (ub - lb);
if any(~isfinite(sigma01)) || any(sigma01 <= 0)
  error("Invalid spec bounds: ub must be > lb for all dims.");
end

mu = mean(XTrain, 1, "omitnan");
sigma = std(XTrain, 0, 1, "omitnan");
sigma(sigma == 0) = 1;

normModel = struct();
normModel.lb = lb;
normModel.ub = ub;
normModel.mu = mu;
normModel.sigma = sigma;
end

