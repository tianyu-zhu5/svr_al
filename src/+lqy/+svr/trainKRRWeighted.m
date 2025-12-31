function base = trainKRRWeighted(ZTrain, yTrain, hp, w)
%TRAINKRRWEIGHTED Weighted kernel ridge regression with RBF kernel (toolbox-free).
%
% Minimizes: sum_i w_i (f(x_i)-y_i)^2 + lambda ||f||^2

arguments
  ZTrain (:,:) double
  yTrain (:,1) double
  hp (1,1) struct
  w (:,1) double
end

if ~isfield(hp, "ell") || ~isfield(hp, "lambda")
  error("hp.ell and hp.lambda are required.");
end

ell = hp.ell;
lambda = hp.lambda;
if ~(isfinite(ell) && ell > 0)
  error("hp.ell must be > 0.");
end
if ~(isfinite(lambda) && lambda > 0)
  error("hp.lambda must be > 0.");
end

w = w(:);
w(~isfinite(w) | w < 0) = 0;
if all(w == 0)
  w(:) = 1;
end

D2 = lqy.util.sqDist(ZTrain, ZTrain);
K = exp(-D2 ./ (2 .* ell.^2));

% Solve: (W*K + lambda*I) * alpha = W*y, where W=diag(w)
A = (w .* K) + lambda .* eye(size(K));
b = w .* yTrain;
alpha = A \ b;

base = struct();
base.type = "krr";
base.ZTrain = ZTrain;
base.alpha = alpha;
base.hp = hp;
base.meta = struct("weighted", true);
end

