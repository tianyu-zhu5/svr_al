function qv = quantile1(x, q)
%QUANTILE1 Toolbox-free quantile for a vector.

arguments
  x (:,1) double
  q (1,1) double {mustBeGreaterThanOrEqual(q,0), mustBeLessThanOrEqual(q,1)}
end

x = x(isfinite(x));
if isempty(x)
  qv = NaN;
  return;
end

x = sort(x);
n = numel(x);
if n == 1
  qv = x(1);
  return;
end

pos = 1 + (n - 1) * q;
lo = floor(pos);
hi = ceil(pos);
if lo == hi
  qv = x(lo);
else
  w = pos - lo;
  qv = (1 - w) * x(lo) + w * x(hi);
end
end

