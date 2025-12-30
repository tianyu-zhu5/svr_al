function D = sqDist(X, Y)
%SQDIST Squared Euclidean distance matrix between rows of X and Y.

arguments
  X (:,:) double
  Y (:,:) double
end

X2 = sum(X.^2, 2);
Y2 = sum(Y.^2, 2).';
D = X2 + Y2 - 2 .* (X * Y.');
D = max(D, 0);
end

