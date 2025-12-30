function X01 = toUnitBox(X, normModel)
%TOUNITBOX Scale physical X into [0,1]^d using normModel.lb/ub.

arguments
  X (:,:) double
  normModel (1,1) struct
end

lb = normModel.lb(:).';
ub = normModel.ub(:).';
den = (ub - lb);

X01 = (X - lb) ./ den;
end

