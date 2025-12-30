function [mu, sigma, yAll] = predictBootstrap(model, XQuery)
%PREDICTBOOTSTRAP Predict mu/sigma from bootstrap ensemble.

arguments
  model (1,1) struct
  XQuery (:,:) double
end

Zq = lqy.norm.zscoreApply(XQuery, model.normModel);

M = numel(model.models);
Nq = size(XQuery, 1);
yAll = zeros(Nq, M);

for m = 1:M
  base = model.models{m};
  if isstruct(base) && isfield(base, "type") && base.type == "krr"
    yAll(:,m) = lqy.svr.predictKRR(base, Zq);
  else
    yAll(:,m) = predict(base, Zq);
  end
end

mu = mean(yAll, 2, "omitnan");
sigma = std(yAll, 0, 2, "omitnan");
sigma(~isfinite(sigma)) = 0;
end

