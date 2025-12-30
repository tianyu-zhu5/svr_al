function idx = weightedResample(p, K)
%WEIGHTEDRESAMPLE Sample indices 1..N with replacement from weights p.

arguments
  p (:,1) double
  K (1,1) double {mustBeInteger, mustBePositive}
end

p = p(:);
if any(p < 0) || ~all(isfinite(p))
  error("Weights p must be finite and >= 0.");
end
if all(p == 0)
  p(:) = 1;
end
p = p ./ sum(p);

edges = cumsum(p);
edges(end) = 1;
r = rand(K, 1);

idx = zeros(K, 1);
for k = 1:K
  idx(k) = find(edges >= r(k), 1, "first");
end
end

