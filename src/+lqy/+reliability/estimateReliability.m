function [Pf, beta] = estimateReliability(mu, cfg)
%ESTIMATERELIABILITY Estimate Pf and beta from surrogate mean.
%
% beta = -Phi^{-1}(Pf)

arguments
  mu (:,1) double
  cfg (1,1) struct
end

Pf = lqy.reliability.estimatePf(mu, cfg);
beta = -lqy.util.norminv1(Pf);
end

