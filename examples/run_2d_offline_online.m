% 2D offline + online refinement demo (M5).

thisFileDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(thisFileDir, "..");
addpath(fullfile(projectRoot, "src"));

cfg = struct();
cfg.seed = 7;
cfg.Npool = 30000;
cfg.N0 = 20;
cfg.budgetTotal = 100;
cfg.bootstrapM = 15;
cfg.weights = struct("tauQuantile", 0.30, "form", "inv");
cfg.stop = struct("useUminStable", true, "UminDelta", 1e-3, "K", 3, "usePfConv", true, "PfEta", 1e-2, "minIters", 12);
cfg.surrogate = struct("method", "auto");
cfg.tune = struct("enabled", true, "method", "auto", "methodHint", "auto", "Kfold", 5, "maxEvals", 12, "everyNeval", 10, "minTrainN", 20, "metric", "wrmse");

% Online (M5) settings
cfg.online = struct();
cfg.online.deltaX = 0.05;     % trigger if design point changes less than this (physical space)
cfg.online.Utrigger = 1.8;    % trigger if Umin in neighborhood is small
cfg.online.alpha = 0.93;      % strong exploitation online
cfg.online.r0 = 0.12;         % initial neighborhood radius in [0,1]^d
cfg.online.rMin = 0.03;
cfg.online.rShrink = 0.85;
cfg.online.maxIters = 12;

rng(cfg.seed, "twister");

spec = struct();
spec.d = 2;
spec.lb = [-3, -3];
spec.ub = [ 3,  3];
spec.dist(1) = struct("type","normal", "params", struct("mu",0, "sigma",1));
spec.dist(2) = struct("type","normal", "params", struct("mu",0, "sigma",1));

spec.trueG = @(X) 1.5 - (X(:,1).^2 + X(:,2).^2);
spec.evalFcn = spec.trueG;

outDir = lqy.io.makeOutDir(fullfile(projectRoot, "out"), "2d_offline_online");

% ----- Offline stage -----
cfg.resume = false;
[model, log, pool] = lqy.al.runOffline(spec, cfg, outDir);

% ----- Online stage (simulate a design point trajectory) -----
% In real use, designHist comes from RBDO iterations.
designHist = [-0.4 -0.2; -0.2 -0.1; -0.1 -0.05; -0.08 -0.04; -0.075 -0.038];

budgetRemain = cfg.budgetTotal - numel(log.poolIdx_all);
[model, log, used, info] = lqy.online.onlineRefine(spec, pool, model, log, designHist, budgetRemain, cfg);
fprintf("Online refine triggered=%d reason=%s usedBudget=%d iters=%d\\n", info.triggered, info.triggerReason, used, info.iters);

stageToSave = "offline";
if used > 0
  stageToSave = "online";
end

lqy.io.saveModelAndLog(model, log, outDir, stageToSave);
lqy.viz.plot2DResults(spec, pool, log, model, outDir);

disp("Offline+Online demo complete.");
disp("Output: " + outDir);
