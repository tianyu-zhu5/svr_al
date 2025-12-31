function [ghat, sigma, yEns] = predictSurrogate(surr, XQuery)
%PREDICTSURROGATE Predict g-hat and sigma for A-mean surrogate.
%
% ghat: from main model only
% sigma: std across bootstrap ensemble predictions

arguments
  surr (1,1) struct
  XQuery (:,:) double
end

Zq = lqy.norm.zscoreApply(XQuery, surr.normModel);
ghat = lqy.surrogate.predictMainModel(surr.main, Zq);

M = numel(surr.ens.models);
Nq = size(XQuery, 1);
yEns = zeros(Nq, M);
for m = 1:M
  base = surr.ens.models{m};
  if isstruct(base) && isfield(base, "type") && base.type == "krr"
    yEns(:,m) = lqy.svr.predictKRR(base, Zq);
  else
    yEns(:,m) = predict(base, Zq);
  end
end

sigma = std(yEns, 0, 2, "omitnan");
sigma(~isfinite(sigma)) = 0;
end

