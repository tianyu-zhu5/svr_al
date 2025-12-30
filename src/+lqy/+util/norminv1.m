function x = norminv1(p)
%NORMINV1 Toolbox-free inverse standard normal CDF.

arguments
  p (:,:) double
end

p = min(max(p, 1e-300), 1 - 1e-16);
x = sqrt(2) .* erfinv(2 .* p - 1);
end

