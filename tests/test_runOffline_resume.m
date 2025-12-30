classdef test_runOffline_resume < matlab.unittest.TestCase
  methods (Test)
    function recordsPoolIdxAndResumes(testCase)
      thisFileDir = fileparts(mfilename("fullpath"));
      projectRoot = fullfile(thisFileDir, "..");
      addpath(fullfile(projectRoot, "src"));

      tmpDir = fullfile(tempdir, "lqy_test_runOffline_" + string(randi(1e9)));
      mkdir(tmpDir);

      spec = struct();
      spec.d = 2;
      spec.lb = [-3 -3];
      spec.ub = [3 3];
      spec.dist(1) = struct("type","normal","params",struct("mu",0,"sigma",1));
      spec.dist(2) = struct("type","normal","params",struct("mu",0,"sigma",1));
      spec.evalFcn = @(X) 1.0 - (X(:,1).^2 + X(:,2).^2);

      cfg = struct();
      cfg.seed = 1;
      cfg.Npool = 2000;
      cfg.N0 = 10;
      cfg.bootstrapM = 5;
      cfg.surrogate = struct("method","krr");
      cfg.tune = struct("enabled", false);
      cfg.stop = struct("useUminStable", false, "usePfConv", false);

      cfg.budgetTotal = cfg.N0 + 2;
      cfg.resume = false;
      [~, log1] = lqy.al.runOffline(spec, cfg, tmpDir);
      testCase.verifyTrue(isfield(log1, "poolIdx_all"));
      testCase.verifyEqual(numel(log1.poolIdx_all), size(log1.X_all,1));
      testCase.verifyEqual(numel(unique(log1.poolIdx_all)), numel(log1.poolIdx_all));

      cfg.budgetTotal = cfg.N0 + 4;
      cfg.resume = true;
      [~, log2] = lqy.al.runOffline(spec, cfg, tmpDir);
      testCase.verifyGreaterThan(numel(log2.poolIdx_all), numel(log1.poolIdx_all));
      testCase.verifyEqual(numel(unique(log2.poolIdx_all)), numel(log2.poolIdx_all));
    end
  end
end

