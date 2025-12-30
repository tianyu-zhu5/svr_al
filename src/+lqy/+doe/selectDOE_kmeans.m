function [X0, idx0] = selectDOE_kmeans(pool, cfg, normModel)
%SELECTDOE_KMEANS Select initial DOE points by k-means centers projected to nearest pool points.

arguments
  pool
  cfg (1,1) struct
  normModel (1,1) struct
end

mustHave(cfg, ["N0"]);

Xpool = pool;
if isstruct(pool)
  if ~isfield(pool, "X"); error("pool.X is required."); end
  Xpool = pool.X;
end

N = size(Xpool, 1);
if cfg.N0 > N
  error("cfg.N0 (%d) cannot exceed pool size (%d).", cfg.N0, N);
end

X01 = lqy.norm.toUnitBox(Xpool, normModel);

if exist("kmeans", "file") == 2
  [~, C] = kmeans(X01, cfg.N0, "Replicates", 3, "MaxIter", 200);
  idx0 = projectCentersToPool(C, X01);
else
  % Toolbox-free fallback: greedy maximin selection in [0,1]^d.
  idx0 = selectMaximin(X01, cfg.N0);
end
X0 = Xpool(idx0, :);
end

function idx = projectCentersToPool(C, Xpool01)
K = size(C, 1);
idx = zeros(K, 1);
used = false(size(Xpool01, 1), 1);

for k = 1:K
  d2 = sum((Xpool01 - C(k,:)).^2, 2);
  [~, order] = sort(d2, "ascend");
  picked = [];
  for t = 1:numel(order)
    candidate = order(t);
    if ~used(candidate)
      picked = candidate;
      break;
    end
  end
  if isempty(picked)
    error("Failed to find a unique pool point for k=%d.", k);
  end
  idx(k) = picked;
  used(picked) = true;
end
end

function mustHave(s, names)
for k = 1:numel(names)
  if ~isfield(s, names(k))
    error("Missing required field: %s", names(k));
  end
end
end

function idx = selectMaximin(X, K)
N = size(X, 1);
if K > N
  error("K must be <= N.");
end

center = 0.5 .* ones(1, size(X, 2));
d2c = sum((X - center).^2, 2);
[~, first] = min(d2c);

idx = zeros(K, 1);
idx(1) = first;
selected = false(N, 1);
selected(first) = true;

minD2 = sum((X - X(first,:)).^2, 2);
for k = 2:K
  minD2(selected) = -inf;
  [~, next] = max(minD2);
  idx(k) = next;
  selected(next) = true;
  d2new = sum((X - X(next,:)).^2, 2);
  minD2 = min(minD2, d2new);
end
end
