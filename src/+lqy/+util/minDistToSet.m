function dmin = minDistToSet(X, Y)
%MINDISTTOSET Min Euclidean distance from each row of X to set Y (row-wise).

arguments
  X (:,:) double
  Y (:,:) double
end

if isempty(Y)
  dmin = inf(size(X,1), 1);
  return;
end

D2 = lqy.util.sqDist(X, Y);
dmin = sqrt(min(D2, [], 2));
end

