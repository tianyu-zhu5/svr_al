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
cfg.surrogate = struct("method", "auto"); % 'auto'|'svr'|'krr'
cfg.tune = struct("enabled", true, "method", "auto", "methodHint", "auto", "Kfold", 5, "maxEvals", 12, "everyNeval", 10, "minTrainN", 20, "metric", "wrmse");

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

resumeDir = string(getenv("LQY_OUTDIR"));
if strlength(resumeDir) > 0 && exist(resumeDir, "dir")
  outDir = resumeDir;
  cfg.resume = true;
else
  outDir = lqy.io.makeOutDir(fullfile(projectRoot, "out"), "2d_offline_mvp");
  cfg.resume = false;
end

[model, log, pool] = lqy.al.runOffline(spec, cfg, outDir);

X0 = log.X_all(log.stage == 0, :);
lqy.viz.plot2DResults(spec, pool, log, outDir);
lqy.viz.plot2DPoolAndDOE(spec, pool, X0, outDir);

disp("MVP complete.");
disp("Output: " + outDir);
