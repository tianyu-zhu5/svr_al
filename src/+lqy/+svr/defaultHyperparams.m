function hp = defaultHyperparams(ZTrain, yTrain, cfg)
%DEFAULTHYPERPARAMS Pick reasonable hyperparameters for toolbox-free surrogate.

arguments
  ZTrain (:,:) double
  yTrain (:,1) double
  cfg (1,1) struct
end

N = size(ZTrain, 1);
if N < 2
  ell = 1;
else
  % Median pairwise distance heuristic.
  idx = randperm(N, min(N, 50));
  D2 = lqy.util.sqDist(ZTrain(idx,:), ZTrain(idx,:));
  D2 = D2(triu(true(size(D2)), 1));
  d = sqrt(max(median(D2, "omitnan"), 1e-12));
  ell = max(d, 1e-3);
end

sy = std(yTrain, 0, 1, "omitnan");
if ~isfinite(sy) || sy == 0
  sy = 1;
end

lambda = 1e-6 * sy^2;
if isfield(cfg, "krr") && isfield(cfg.krr, "lambda")
  lambda = cfg.krr.lambda;
end
if isfield(cfg, "krr") && isfield(cfg.krr, "ell")
  ell = cfg.krr.ell;
end

hp = struct("ell", ell, "lambda", lambda);
end

