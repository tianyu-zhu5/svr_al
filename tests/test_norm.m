classdef test_norm < matlab.unittest.TestCase
  methods (Test)
    function unitBoxRoundTrip(testCase)
      thisFileDir = fileparts(mfilename("fullpath"));
      projectRoot = fullfile(thisFileDir, "..");
      addpath(fullfile(projectRoot, "src"));
      spec = struct("lb", [-2, 0], "ub", [2, 10]);
      X = [0, 5; -2, 0; 2, 10];
      nm = struct("lb", spec.lb, "ub", spec.ub);
      X01 = lqy.norm.toUnitBox(X, nm);
      X2 = lqy.norm.fromUnitBox(X01, nm);
      testCase.verifyEqual(X2, X, "AbsTol", 1e-12);
    end

    function zscoreApplyUsesFrozenParams(testCase)
      thisFileDir = fileparts(mfilename("fullpath"));
      projectRoot = fullfile(thisFileDir, "..");
      addpath(fullfile(projectRoot, "src"));
      X = [1 2; 3 4; 5 6];
      spec = struct("lb", [0 0], "ub", [10 10]);
      nm = lqy.norm.fitNormModel(X, spec);
      Z = lqy.norm.zscoreApply(X, nm);
      testCase.verifySize(Z, size(X));
      testCase.verifyEqual(mean(Z,1), [0 0], "AbsTol", 1e-12);
    end
  end
end
