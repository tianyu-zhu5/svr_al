function model = trainBootstrapSurrogate(XTrain, gTrain, cfg, normModel)
%TRAINBOOTSTRAPSURROGATE Bootstrap ensemble surrogate that outputs mu/sigma.
%
% If fitrsvm is available, uses SVR; otherwise uses toolbox-free KRR.

arguments
  XTrain (:,:) double
  gTrain (:,1) double
  cfg (1,1) struct
  normModel (1,1) struct
end

M = 20;
if isfield(cfg, "bootstrapM")
  M = cfg.bootstrapM;
end

ZTrain = lqy.norm.zscoreApply(XTrain, normModel);
hp = lqy.svr.defaultHyperparams(ZTrain, gTrain, cfg);

w = lqy.svr.computeWeights(gTrain, cfg);
p = w ./ sum(w);

N = size(XTrain, 1);
models = cell(M, 1);
theta = cell(M, 1);

useSVR = exist("fitrsvm", "file") == 2;
for m = 1:M
  idx = lqy.util.weightedResample(p(:), N);
  Zb = ZTrain(idx, :);
  yb = gTrain(idx);

  if useSVR
    eps0 = 0.1 * std(yb, 0, 1, "omitnan");
    if ~isfinite(eps0) || eps0 <= 0; eps0 = 1e-3; end
    svm = fitrsvm(Zb, yb, ...
      "KernelFunction", "gaussian", ...
      "KernelScale", hp.ell, ...
      "BoxConstraint", 10, ...
      "Epsilon", eps0, ...
      "Standardize", false);
    models{m} = svm;
    theta{m} = struct("type","svr","KernelScale",hp.ell,"BoxConstraint",10,"Epsilon",eps0);
  else
    base = lqy.svr.trainKRR(Zb, yb, hp);
    models{m} = base;
    theta{m} = struct("type","krr","ell",hp.ell,"lambda",hp.lambda);
  end
end

model = struct();
model.models = models;
model.theta = theta;
model.normModel = normModel;
model.meta = struct("bootstrapM", M, "useSVR", useSVR);
end
