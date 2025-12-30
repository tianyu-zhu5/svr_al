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
      cfg = struct("bootstrapM", 7, "weights", struct("form","inv","tauQuantile",0.3));
      model = lqy.svr.trainBootstrapSurrogate(X, g, cfg, nm);

      Xq = [0 0; 1 1; -1 0.5];
      [mu, sigma] = lqy.svr.predictBootstrap(model, Xq);
      testCase.verifySize(mu, [3 1]);
      testCase.verifySize(sigma, [3 1]);
      testCase.verifyGreaterThanOrEqual(min(sigma), 0);
    end
  end
end

