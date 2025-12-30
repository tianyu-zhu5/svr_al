function pool = buildPoolTruncatedLHS(spec, cfg)
%BUILDPOOLTRUNCATEDLHS Build a fixed candidate pool using truncated inverse-CDF + LHS.

arguments
  spec (1,1) struct
  cfg (1,1) struct
end

mustHave(spec, ["d","lb","ub","dist"]);
mustHave(cfg, ["Npool"]);

d = spec.d;
lb = spec.lb(:).';
ub = spec.ub(:).';

if numel(lb) ~= d || numel(ub) ~= d
  error("spec.lb/spec.ub must be 1xd where d=spec.d.");
end
if numel(spec.dist) ~= d
  error("spec.dist must have length d.");
end

N = cfg.Npool;
if ~isscalar(N) || N <= 0
  error("cfg.Npool must be a positive scalar.");
end

R = sampleLHS(N, d);

X = zeros(N, d);
for j = 1:d
  dist = lqy.dist.makeDist(spec.dist(j));
  Plb = dist.cdf(lb(j));
  Pub = dist.cdf(ub(j));
  if ~(isfinite(Plb) && isfinite(Pub) && Pub > Plb)
    error("Invalid truncation probabilities for dim %d (Plb=%g, Pub=%g).", j, Plb, Pub);
  end
  u = Plb + (Pub - Plb) .* R(:,j);
  u = min(max(u, 1e-12), 1-1e-12);
  X(:,j) = dist.icdf(u);
end

pool = struct();
pool.X = X;
pool.label = repmat("prob", N, 1);
pool.meta = struct("method","truncated-icdf-lhs","N",N,"d",d);
end

function R = sampleLHS(N, d)
if exist("lhsdesign", "file") == 2
  R = lhsdesign(N, d, "smooth", "off");
else
  % Fallback: not a true LHS, but keeps the pipeline runnable.
  R = rand(N, d);
end
end

function mustHave(s, names)
for k = 1:numel(names)
  if ~isfield(s, names(k))
    error("Missing required field: %s", names(k));
  end
end
end
