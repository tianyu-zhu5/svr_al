function idx = localPoolIndices(poolX, xd, r, normModel)
%LOCALPOOLINDICES Indices of pool points within radius r of design point in unit box space.

arguments
  poolX (:,:) double
  xd (1,:) double
  r (1,1) double {mustBePositive}
  normModel (1,1) struct
end

X01 = lqy.norm.toUnitBox(poolX, normModel);
xd01 = lqy.norm.toUnitBox(xd, normModel);
d = sqrt(sum((X01 - xd01).^2, 2));
idx = find(d < r);
end

