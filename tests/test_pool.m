classdef test_pool < matlab.unittest.TestCase
  methods (Test)
    function poolWithinBounds(testCase)
      thisFileDir = fileparts(mfilename("fullpath"));
      projectRoot = fullfile(thisFileDir, "..");
      addpath(fullfile(projectRoot, "src"));
      cfg = struct("Npool", 2000);
      spec = struct();
      spec.d = 2;
      spec.lb = [-3, -3];
      spec.ub = [ 3,  3];
      spec.dist(1) = struct("type","normal", "params", struct("mu",0, "sigma",1));
      spec.dist(2) = struct("type","normal", "params", struct("mu",0, "sigma",1));

      rng(1);
      pool = lqy.pool.buildPoolTruncatedLHS(spec, cfg);
      testCase.verifyGreaterThanOrEqual(min(pool.X, [], 1), spec.lb - 1e-10);
      testCase.verifyLessThanOrEqual(max(pool.X, [], 1), spec.ub + 1e-10);
    end
  end
end
