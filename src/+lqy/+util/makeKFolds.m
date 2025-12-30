function foldId = makeKFolds(N, K)
%MAKEKFOLDS Create deterministic-ish K-fold assignments without toolboxes.

arguments
  N (1,1) double {mustBeInteger, mustBePositive}
  K (1,1) double {mustBeInteger, mustBePositive}
end

K = min(K, N);
perm = randperm(N);
foldId = zeros(N, 1);
for i = 1:N
  foldId(perm(i)) = mod(i-1, K) + 1;
end
end

