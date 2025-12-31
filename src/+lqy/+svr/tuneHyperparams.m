function [hpBest, info] = tuneHyperparams(ZTrain, yTrain, w, cfg, methodHint)
%TUNEHYPERPARAMS Hyperparameter tuning using OOF (K-fold) metric.
%
% Supports:
% - SVR (fitrsvm) + bayesopt/random
% - KRR (toolbox-free) + bayesopt/random (bayesopt optional)
%
% methodHint: "auto"|"svr"|"krr"

arguments
  ZTrain (:,:) double
  yTrain (:,1) double
  w (:,1) double
  cfg (1,1) struct
  methodHint (1,1) string = "auto"
end

K = 5;
maxEvals = 20;
method = "auto";
metric = "wrmse";

if isfield(cfg, "tune") && ~isempty(cfg.tune)
  if isfield(cfg.tune, "Kfold"); K = cfg.tune.Kfold; end
  if isfield(cfg.tune, "maxEvals"); maxEvals = cfg.tune.maxEvals; end
  if isfield(cfg.tune, "method"); method = string(cfg.tune.method); end
  if isfield(cfg.tune, "metric"); metric = string(cfg.tune.metric); end
end

useSVR = false;
if methodHint == "svr"
  useSVR = exist("fitrsvm", "file") == 2;
elseif methodHint == "krr"
  useSVR = false;
else
  % auto
  useSVR = exist("fitrsvm", "file") == 2;
end

canBayesopt = exist("bayesopt", "file") == 2 && exist("optimizableVariable", "file") == 2;
useBayesopt = (lower(method) == "bayesopt") || (lower(method) == "auto" && canBayesopt);

K = min(K, size(ZTrain,1));
foldId = lqy.util.makeKFolds(size(ZTrain,1), K);

if useSVR
  % Ranges in z-scored space
  ksRange = [log(0.05), log(5)];
  bcRange = [log(1e-2), log(1e3)];
  epsRange = [log(1e-4), log(0.5)];

  if useBayesopt
    vars = [ ...
      optimizableVariable("logKernelScale", ksRange), ...
      optimizableVariable("logBoxConstraint", bcRange), ...
      optimizableVariable("logEpsilon", epsRange) ...
    ];
    obj = @(T) svrObjective(T, ZTrain, yTrain, w, foldId, metric);
    results = bayesopt(obj, vars, ...
      "MaxObjectiveEvaluations", maxEvals, ...
      "IsObjectiveDeterministic", true, ...
      "Verbose", 0, ...
      "PlotFcn", []);
    best = results.XAtMinObjective;
    hpBest = struct( ...
      "type","svr", ...
      "KernelScale", exp(best.logKernelScale), ...
      "BoxConstraint", exp(best.logBoxConstraint), ...
      "Epsilon", exp(best.logEpsilon));
    info = struct("method","bayesopt","bestObjective",results.MinObjective,"useSVR",true);
  else
    hpBest = randomSearchSVR(ZTrain, yTrain, w, foldId, metric, maxEvals, ksRange, bcRange, epsRange);
    info = struct("method","random","bestObjective",hpBest.bestObjective,"useSVR",true);
    hpBest = rmfield(hpBest, "bestObjective");
  end
else
  ellRange = [log(0.05), log(5)];
  lamRange = [log(1e-12), log(1e-3)];

  if useBayesopt
    vars = [ ...
      optimizableVariable("logEll", ellRange), ...
      optimizableVariable("logLambda", lamRange) ...
    ];
    obj = @(T) krrObjective(T, ZTrain, yTrain, w, foldId, metric);
    results = bayesopt(obj, vars, ...
      "MaxObjectiveEvaluations", maxEvals, ...
      "IsObjectiveDeterministic", true, ...
      "Verbose", 0, ...
      "PlotFcn", []);
    best = results.XAtMinObjective;
    hpBest = struct("type","krr", "ell", exp(best.logEll), "lambda", exp(best.logLambda));
    info = struct("method","bayesopt","bestObjective",results.MinObjective,"useSVR",false);
  else
    hpBest = randomSearchKRR(ZTrain, yTrain, w, foldId, metric, maxEvals, ellRange, lamRange);
    info = struct("method","random","bestObjective",hpBest.bestObjective,"useSVR",false);
    hpBest = rmfield(hpBest, "bestObjective");
  end
end
end

function out = svrObjective(T, Z, y, w, foldId, metric)
ks = exp(T.logKernelScale);
bc = exp(T.logBoxConstraint);
eps0 = exp(T.logEpsilon);
out = oofScoreSVR(Z, y, w, foldId, metric, ks, bc, eps0);
end

function out = krrObjective(T, Z, y, w, foldId, metric)
ell = exp(T.logEll);
lambda = exp(T.logLambda);
out = oofScoreKRR(Z, y, w, foldId, metric, ell, lambda);
end

function hp = randomSearchSVR(Z, y, w, foldId, metric, maxEvals, ksRange, bcRange, epsRange)
bestObj = inf;
best = struct("KernelScale",1,"BoxConstraint",10,"Epsilon",1e-3);
for i = 1:maxEvals
  ks = exp(ksRange(1) + rand()*(ksRange(2)-ksRange(1)));
  bc = exp(bcRange(1) + rand()*(bcRange(2)-bcRange(1)));
  eps0 = exp(epsRange(1) + rand()*(epsRange(2)-epsRange(1)));
  obj = oofScoreSVR(Z, y, w, foldId, metric, ks, bc, eps0);
  if obj < bestObj
    bestObj = obj;
    best = struct("KernelScale", ks, "BoxConstraint", bc, "Epsilon", eps0);
  end
end
hp = struct("type","svr", "KernelScale", best.KernelScale, "BoxConstraint", best.BoxConstraint, "Epsilon", best.Epsilon, "bestObjective", bestObj);
end

function hp = randomSearchKRR(Z, y, w, foldId, metric, maxEvals, ellRange, lamRange)
bestObj = inf;
best = struct("ell",1,"lambda",1e-6);
for i = 1:maxEvals
  ell = exp(ellRange(1) + rand()*(ellRange(2)-ellRange(1)));
  lambda = exp(lamRange(1) + rand()*(lamRange(2)-lamRange(1)));
  obj = oofScoreKRR(Z, y, w, foldId, metric, ell, lambda);
  if obj < bestObj
    bestObj = obj;
    best = struct("ell", ell, "lambda", lambda);
  end
end
hp = struct("type","krr", "ell", best.ell, "lambda", best.lambda, "bestObjective", bestObj);
end

function obj = oofScoreSVR(Z, y, w, foldId, metric, ks, bc, eps0)
K = max(foldId);
pred = nan(size(y));
for k = 1:K
  testMask = foldId == k;
  trainMask = ~testMask;
  if nnz(trainMask) < 5 || nnz(testMask) < 1
    continue;
  end
  mdl = fitrsvm(Z(trainMask,:), y(trainMask), ...
    "KernelFunction", "gaussian", ...
    "KernelScale", ks, ...
    "BoxConstraint", bc, ...
    "Epsilon", eps0, ...
    "Standardize", false, ...
    "Weights", w(trainMask));
  pred(testMask) = predict(mdl, Z(testMask,:));
end
obj = scoreFromPred(y, pred, w, metric);
end

function obj = oofScoreKRR(Z, y, w, foldId, metric, ell, lambda)
K = max(foldId);
pred = nan(size(y));
for k = 1:K
  testMask = foldId == k;
  trainMask = ~testMask;
  if nnz(trainMask) < 5 || nnz(testMask) < 1
    continue;
  end
  base = lqy.svr.trainKRRWeighted(Z(trainMask,:), y(trainMask), struct("ell",ell,"lambda",lambda), w(trainMask));
  pred(testMask) = lqy.svr.predictKRR(base, Z(testMask,:));
end
obj = scoreFromPred(y, pred, w, metric);
end

function obj = scoreFromPred(y, pred, w, metric)
ok = isfinite(pred);
if nnz(ok) < max(5, floor(0.5*numel(y)))
  obj = inf;
  return;
end
yy = y(ok);
pp = pred(ok);
ww = w(ok);
switch lower(metric)
  case "wrmse"
    obj = lqy.util.weightedRmse(yy, pp, ww);
  otherwise
    obj = lqy.util.weightedRmse(yy, pp, ww);
end
end
