function yhat = predictKRR(base, ZQuery)
%PREDICTKRR Predict with KRR base model.

arguments
  base (1,1) struct
  ZQuery (:,:) double
end

ell = base.hp.ell;
D2 = lqy.util.sqDist(ZQuery, base.ZTrain);
Kq = exp(-D2 ./ (2 .* ell.^2));
yhat = Kq * base.alpha;
end

