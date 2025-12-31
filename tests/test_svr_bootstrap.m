classdef test_svr_bootstrap < matlab.unittest.TestCase
  methods (Test)
    function predictsMuSigma(testCase)
      thisFileDir = fileparts(mfilename("fullpath"));
      projectRoot = fullfile(thisFileDir, "..");
      addpath(fullfile(projectRoot, "src"));

      rng(1);
      spec = struct("lb", [-3 -3], "ub", [3 3]);
      X = randn(40,2);
      X = max(min(X, spec.ub), spec.lb);
      g = 1.0 - (X(:,1).^2 + 0.5*X(:,2).^2);

      nm = lqy.norm.fitNormModel(X, struct("lb",spec.lb,"ub",spec.ub));
      cfg = struct();
      cfg.bootstrapM = 7;
      cfg.weights = struct("form","inv","tauQuantile",0.3);
      cfg.surrogate = struct("method","krr");
      cfg.tune = struct("enabled", false);

      base = lqy.svr.defaultHyperparams(lqy.norm.zscoreApply(X, nm), g, cfg);
      hp = struct("type","krr","ell",base.ell,"lambda",base.lambda);
      surr = lqy.surrogate.trainSurrogate(X, g, cfg, struct("lb",spec.lb,"ub",spec.ub), nm, hp);

      Xq = [0 0; 1 1; -1 0.5];
      [ghat, sigma] = lqy.surrogate.predictSurrogate(surr, Xq);
      testCase.verifySize(ghat, [3 1]);
      testCase.verifySize(sigma, [3 1]);
      testCase.verifyGreaterThanOrEqual(min(sigma), 0);

      % A-mean: ghat must come from main model, not ensemble average.
      ghat2 = lqy.surrogate.predictMainModel(surr.main, lqy.norm.zscoreApply(Xq, surr.normModel));
      testCase.verifyEqual(ghat, ghat2, "AbsTol", 1e-12);
    end
  end
end
