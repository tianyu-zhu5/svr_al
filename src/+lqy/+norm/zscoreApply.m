function Z = zscoreApply(X, normModel)
%ZSCOREAPPLY Apply z-score standardization using frozen mu/sigma.

arguments
  X (:,:) double
  normModel (1,1) struct
end

Z = (X - normModel.mu) ./ normModel.sigma;
end

