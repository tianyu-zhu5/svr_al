classdef test_online_trigger < matlab.unittest.TestCase
  methods (Test)
    function triggerByDesignStability(testCase)
      thisFileDir = fileparts(mfilename("fullpath"));
      projectRoot = fullfile(thisFileDir, "..");
      addpath(fullfile(projectRoot, "src"));

      spec = struct("d",2,"lb",[-3 -3],"ub",[3 3]);
      cfg = struct();
      cfg.Npool = 500;
      cfg.online = struct("deltaX", 0.1, "r0", 0.2, "Utrigger", 2.0);

      rng(1);
      spec.dist(1) = struct("type","normal","params",struct("mu",0,"sigma",1));
      spec.dist(2) = struct("type","normal","params",struct("mu",0,"sigma",1));
      spec.evalFcn = @(X) 1.0 - sum(X.^2,2);

      pool = lqy.pool.buildPoolTruncatedLHS(spec, cfg);
      XTrain = pool.X(1:15,:);
      gTrain = spec.evalFcn(XTrain);
      nm = lqy.norm.fitNormModel(XTrain, spec);
      cfg2 = cfg;
      cfg2.bootstrapM = 5;
      cfg2.weights = struct("form","inv","tauQuantile",0.3);
      cfg2.surrogate = struct("method","krr");
      cfg2.tune = struct("enabled", false);
      base = lqy.svr.defaultHyperparams(lqy.norm.zscoreApply(XTrain, nm), gTrain, cfg2);
      hp = struct("type","krr","ell",base.ell,"lambda",base.lambda);
      model = lqy.surrogate.trainSurrogate(XTrain, gTrain, cfg2, spec, nm, hp);

      designHist = [0 0; 0.02 0.01];
      [tr, reason] = lqy.online.shouldTrigger(designHist, model, pool.X, XTrain, cfg);
      testCase.verifyTrue(tr);
      testCase.verifyEqual(string(reason), "design_stable");
    end
  end
end
