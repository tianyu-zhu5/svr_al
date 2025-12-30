function X = fromUnitBox(X01, normModel)
%FROMUNITBOX Inverse of toUnitBox.

arguments
  X01 (:,:) double
  normModel (1,1) struct
end

lb = normModel.lb(:).';
ub = normModel.ub(:).';
X = X01 .* (ub - lb) + lb;
end

