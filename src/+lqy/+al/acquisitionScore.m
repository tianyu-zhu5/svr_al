function [score, detail] = acquisitionScore(ghat, sigma, XQuery, XTrain, cfg, normModel, t)
%ACQUISITIONSCORE Combined exploitation (inverse U) and exploration (min distance).
%
% score: larger is better.

arguments
  ghat (:,1) double
  sigma (:,1) double
  XQuery (:,:) double
  XTrain (:,:) double
  cfg (1,1) struct
  normModel (1,1) struct
  t (1,1) double {mustBeInteger, mustBeNonnegative}
end

epsSigma = 1e-12;
U = abs(ghat) ./ max(sigma, epsSigma);

exploitation = 1 ./ (U + 1e-6); % larger near boundary & uncertain

Xq01 = lqy.norm.toUnitBox(XQuery, normModel);
Xt01 = lqy.norm.toUnitBox(XTrain, normModel);
exploration = lqy.util.minDistToSet(Xq01, Xt01);

alpha = lqy.al.alphaSchedule(t, cfg);
exploration = exploration ./ max(median(exploration, "omitnan"), 1e-12);

score = alpha .* exploitation + (1 - alpha) .* exploration;

detail = struct("U", U, "exploitation", exploitation, "exploration", exploration, "alpha", alpha);
end
