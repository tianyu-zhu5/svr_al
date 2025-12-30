function base = trainKRR(ZTrain, yTrain, hp)
%TRAINKRR Kernel ridge regression with RBF kernel (toolbox-free).

arguments
  ZTrain (:,:) double
  yTrain (:,1) double
  hp (1,1) struct
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

D2 = lqy.util.sqDist(ZTrain, ZTrain);
K = exp(-D2 ./ (2 .* ell.^2));
alpha = (K + lambda .* eye(size(K))) \ yTrain;

base = struct();
base.type = "krr";
base.ZTrain = ZTrain;
base.alpha = alpha;
base.hp = hp;
end

