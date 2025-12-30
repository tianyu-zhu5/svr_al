% 2D offline MVP: build pool -> select DOE -> evaluate true g -> save log and plots.

thisFileDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(thisFileDir, "..");
addpath(fullfile(projectRoot, "src"));

cfg = struct();
cfg.seed = 42;
cfg.Npool = 20000;
cfg.N0 = 20;
cfg.budgetTotal = 80;
cfg.bootstrapM = 15;
cfg.weights = struct("tauQuantile", 0.30, "form", "inv");
cfg.stop = struct("useUminStable", true, "UminDelta", 1e-3, "K", 3, "usePfConv", true, "PfEta", 1e-2);
cfg.stop.minIters = 12;

rng(cfg.seed, "twister");

spec = struct();
spec.d = 2;

% Two independent truncated standard normals on [-3,3].
spec.lb = [-3, -3];
spec.ub = [ 3,  3];
spec.dist(1) = struct("type","normal", "params", struct("mu",0, "sigma",1));
spec.dist(2) = struct("type","normal", "params", struct("mu",0, "sigma",1));

% Analytic limit state (toy): circle boundary.
spec.trueG = @(X) 1.5 - (X(:,1).^2 + X(:,2).^2);
spec.evalFcn = spec.trueG;

outDir = lqy.io.makeOutDir(fullfile(projectRoot, "out"), "2d_offline_mvp");

pool = lqy.pool.buildPoolTruncatedLHS(spec, cfg);
normModelTmp = struct("lb", spec.lb, "ub", spec.ub);
[X0, idx0] = lqy.doe.selectDOE_kmeans(pool, cfg, normModelTmp);

g0 = spec.evalFcn(X0);

log = struct();
log.X_all = X0;
log.g_all = g0;
log.stage = zeros(size(X0,1), 1); % 0=DOE
log.meta = struct("seed", cfg.seed, "cfg", cfg, "spec", rmfield(spec, ["evalFcn","trueG"]));

% Offline active learning loop (MVP)
XTrain = X0;
gTrain = g0;
poolX = pool.X;
picked = false(size(poolX,1), 1);
picked(idx0) = true;

Pf_hat = [];
UminHist = [];
alphaHist = [];
NevalHist = [];

while true
  normModel = lqy.norm.fitNormModel(XTrain, spec);
  model = lqy.svr.trainBootstrapSurrogate(XTrain, gTrain, cfg, normModel);
  [mu, sigma] = lqy.svr.predictBootstrap(model, poolX);

  Pf = lqy.reliability.estimatePf(mu, cfg);
  Pf_hat(end+1,1) = Pf; %#ok<SAGROW>

  U = abs(mu) ./ max(sigma, 1e-12);
  UminHist(end+1,1) = min(U); %#ok<SAGROW>

  t = numel(Pf_hat);
  [score, detail] = lqy.al.acquisitionScore(mu, sigma, poolX, XTrain, cfg, normModel, t);
  alphaHist(end+1,1) = detail.alpha; %#ok<SAGROW>

  score(picked) = -inf;
  [~, nextIdx] = max(score);
  xNext = poolX(nextIdx, :);
  gNext = spec.evalFcn(xNext);

  XTrain = [XTrain; xNext]; %#ok<AGROW>
  gTrain = [gTrain; gNext]; %#ok<AGROW>

  picked(nextIdx) = true;
  log.X_all = XTrain;
  log.g_all = gTrain;
  log.stage = [log.stage; 1]; %#ok<AGROW>

  Neval = size(XTrain, 1);
  NevalHist(end+1,1) = Neval; %#ok<SAGROW>

  state = struct("Neval", Neval, "PfHist", Pf_hat, "UminHist", UminHist);
  [stop, reason] = lqy.al.stopCriteria(state, cfg);
  if stop
    fprintf("Stop: %s (Neval=%d)\\n", reason, Neval);
    break;
  end
end

log.Pf_hat = Pf_hat;
log.Umin = UminHist;
log.alpha = alphaHist;
log.NevalHist = NevalHist;

lqy.io.saveModelAndLog(model, log, outDir, "offline");
lqy.viz.plot2DResults(spec, pool, log, outDir);
lqy.viz.plot2DPoolAndDOE(spec, pool, X0, outDir);

disp("MVP complete.");
disp("Output: " + outDir);
