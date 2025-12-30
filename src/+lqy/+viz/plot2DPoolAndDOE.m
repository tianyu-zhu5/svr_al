function plot2DPoolAndDOE(spec, pool, X0, outDir)
%PLOT2DPOOLANDDOE Visualize pool and DOE for 2D problems.

arguments
  spec (1,1) struct
  pool (1,1) struct
  X0 (:,2) double
  outDir (1,1) string
end

if spec.d ~= 2
  error("plot2DPoolAndDOE only supports d=2.");
end

f = figure("Visible", "off");
scatter(pool.X(:,1), pool.X(:,2), 6, [0.6 0.6 0.6], "filled"); hold on;
scatter(X0(:,1), X0(:,2), 40, "r", "filled");
grid on; box on;
xlabel("x1"); ylabel("x2");
title("Pool and initial DOE");
legend({"pool","DOE"}, "Location", "best");
saveas(f, fullfile(outDir, "pool_doe.png"));
close(f);

if isfield(spec, "trueG") && ~isempty(spec.trueG)
  f = figure("Visible", "off");
  x1 = linspace(spec.lb(1), spec.ub(1), 200);
  x2 = linspace(spec.lb(2), spec.ub(2), 200);
  [X1, X2] = meshgrid(x1, x2);
  G = spec.trueG([X1(:), X2(:)]);
  G = reshape(G, size(X1));
  contour(X1, X2, G, [0 0], "k", "LineWidth", 1.5); hold on;
  scatter(X0(:,1), X0(:,2), 40, "r", "filled");
  scatter(pool.X(:,1), pool.X(:,2), 6, [0.8 0.8 0.8], "filled");
  grid on; box on;
  xlabel("x1"); ylabel("x2");
  title("True limit state g(x)=0 with samples");
  legend({"g(x)=0","DOE","pool"}, "Location", "best");
  saveas(f, fullfile(outDir, "true_boundary_samples.png"));
  close(f);
end
end

