function dist = makeDist(distSpec)
%MAKEDIST Create distribution helpers without requiring toolboxes.
%
% Returns a struct with:
%   - type, params
%   - cdf(x)  : function handle
%   - icdf(u) : function handle
%
% distSpec fields:
%   - type: 'normal'|'lognormal'|'uniform' (extend as needed)
%   - params: struct with distribution parameters
%   - (optional) dist: a MATLAB probability distribution object that supports cdf/icdf

arguments
  distSpec (1,1) struct
end

if isfield(distSpec, "dist") && ~isempty(distSpec.dist)
  d = distSpec.dist;
  dist = struct();
  dist.type = "object";
  dist.params = struct();
  dist.cdf = @(x) cdf(d, x);
  dist.icdf = @(u) icdf(d, u);
  return;
end

if ~isfield(distSpec, "type")
  error("distSpec.type is required.");
end

type = lower(string(distSpec.type));
params = struct();
if isfield(distSpec, "params") && ~isempty(distSpec.params)
  params = distSpec.params;
end

dist = struct();
dist.type = type;
dist.params = params;

switch type
  case "normal"
    mustHave(params, ["mu","sigma"]);
    mu = params.mu;
    sigma = params.sigma;
    if ~(isfinite(mu) && isfinite(sigma) && sigma > 0)
      error("Normal params must satisfy isfinite(mu) and sigma>0.");
    end
    dist.cdf = @(x) normalCdf(x, mu, sigma);
    dist.icdf = @(u) normalIcdf(u, mu, sigma);
  case "lognormal"
    % MATLAB's Lognormal uses underlying Normal(mu,sigma) on log(X).
    mustHave(params, ["mu","sigma"]);
    mu = params.mu;
    sigma = params.sigma;
    if ~(isfinite(mu) && isfinite(sigma) && sigma > 0)
      error("Lognormal params must satisfy isfinite(mu) and sigma>0.");
    end
    dist.cdf = @(x) lognormalCdf(x, mu, sigma);
    dist.icdf = @(u) exp(normalIcdf(u, mu, sigma));
  case "uniform"
    if isfield(params,"a") && isfield(params,"b")
      a = params.a; b = params.b;
    else
      mustHave(params, ["lower","upper"]);
      a = params.lower; b = params.upper;
    end
    if ~(isfinite(a) && isfinite(b) && b > a)
      error("Uniform params must satisfy upper>lower.");
    end
    dist.cdf = @(x) uniformCdf(x, a, b);
    dist.icdf = @(u) a + (b - a) .* u;
  otherwise
    error("Unsupported dist type: %s", type);
end
end

function mustHave(s, names)
for k = 1:numel(names)
  if ~isfield(s, names(k))
    error("distSpec.params.%s is required.", names(k));
  end
end
end

function p = normalCdf(x, mu, sigma)
z = (x - mu) ./ (sigma * sqrt(2));
p = 0.5 .* (1 + erf(z));
p = min(max(p, 0), 1);
end

function x = normalIcdf(u, mu, sigma)
u = min(max(u, 0), 1);
x = mu + sigma .* sqrt(2) .* erfinv(2 .* u - 1);
end

function p = uniformCdf(x, a, b)
p = (x - a) ./ (b - a);
p = min(max(p, 0), 1);
end

function p = lognormalCdf(x, mu, sigma)
p = zeros(size(x));
pos = x > 0;
p(pos) = normalCdf(log(x(pos)), mu, sigma);
end
