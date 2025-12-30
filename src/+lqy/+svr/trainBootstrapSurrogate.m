function model = trainBootstrapSurrogate(XTrain, gTrain, cfg, normModel, hp)
%TRAINBOOTSTRAPSURROGATE Bootstrap ensemble surrogate that outputs mu/sigma.
%
% If fitrsvm is available, uses SVR; otherwise uses toolbox-free KRR.

arguments
  XTrain (:,:) double
  gTrain (:,1) double
  cfg (1,1) struct
  normModel (1,1) struct
  hp = []
end

M = 20;
if isfield(cfg, "bootstrapM")
  M = cfg.bootstrapM;
end

ZTrain = lqy.norm.zscoreApply(XTrain, normModel);
hpDefault = lqy.svr.defaultHyperparams(ZTrain, gTrain, cfg);
if isempty(hp)
  hp = hpDefault;
end

w = lqy.svr.computeWeights(gTrain, cfg);
p = w ./ sum(w);

N = size(XTrain, 1);
models = cell(M, 1);
theta = cell(M, 1);

forceMethod = "auto";
if isfield(cfg, "surrogate") && isfield(cfg.surrogate, "method")
  forceMethod = string(cfg.surrogate.method);
end

useSVR = exist("fitrsvm", "file") == 2 && (forceMethod == "auto" || forceMethod == "svr");
if forceMethod == "krr"
  useSVR = false;
end

for m = 1:M
  idx = lqy.util.weightedResample(p(:), N);
  Zb = ZTrain(idx, :);
  yb = gTrain(idx);

  if useSVR
    if isfield(hp, "KernelScale")
      ks = hp.KernelScale;
    else
      ks = hpDefault.ell;
    end
    bc = 10;
    if isfield(hp, "BoxConstraint"); bc = hp.BoxConstraint; end
    eps0 = 0.1 * std(yb, 0, 1, "omitnan");
    if isfield(hp, "Epsilon"); eps0 = hp.Epsilon; end
    if ~isfinite(eps0) || eps0 <= 0; eps0 = 1e-3; end
    svm = fitrsvm(Zb, yb, ...
      "KernelFunction", "gaussian", ...
      "KernelScale", ks, ...
      "BoxConstraint", bc, ...
      "Epsilon", eps0, ...
      "Standardize", false);
    models{m} = svm;
    theta{m} = struct("type","svr","KernelScale",ks,"BoxConstraint",bc,"Epsilon",eps0);
  else
    if ~isfield(hp, "ell"); hp.ell = hpDefault.ell; end
    if ~isfield(hp, "lambda"); hp.lambda = hpDefault.lambda; end
    base = lqy.svr.trainKRR(Zb, yb, struct("ell",hp.ell,"lambda",hp.lambda));
    models{m} = base;
    theta{m} = struct("type","krr","ell",hp.ell,"lambda",hp.lambda);
  end
end

model = struct();
model.models = models;
model.theta = theta;
model.hpShared = hp;
model.normModel = normModel;
model.meta = struct("bootstrapM", M, "useSVR", useSVR);
end
