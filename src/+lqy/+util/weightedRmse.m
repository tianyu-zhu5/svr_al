function v = weightedRmse(yTrue, yPred, w)
%WEIGHTEDRMSE Compute weighted RMSE.

arguments
  yTrue (:,1) double
  yPred (:,1) double
  w (:,1) double
end

e = yPred - yTrue;
w = w(:);
w(~isfinite(w) | w < 0) = 0;
if all(w == 0)
  w(:) = 1;
end

v = sqrt(sum(w .* (e.^2)) ./ sum(w));
end

