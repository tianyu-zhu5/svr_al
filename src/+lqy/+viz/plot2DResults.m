function plot2DResults(spec, pool, log, outDir)
%PLOT2DRESULTS Plot Pf convergence and 2D sampling trajectory.

arguments
  spec (1,1) struct
  pool (1,1) struct
  log (1,1) struct
  outDir (1,1) string
end

if spec.d ~= 2
  return;
end

if isfield(log, "Pf_hat") && ~isempty(log.Pf_hat)
  f = figure("Visible", "off");
  plot(log.NevalHist, log.Pf_hat, "-o", "LineWidth", 1.2); grid on; box on;
  xlabel("累计 evaluator 调用次数");
  ylabel("P_f 估计");
  title("Pf 收敛曲线（surrogate mean）");
  saveas(f, fullfile(outDir, "pf_convergence.png"));
  close(f);
end

f = figure("Visible", "off");
hold on;
scatter(pool.X(:,1), pool.X(:,2), 6, [0.85 0.85 0.85], "filled");

X = log.X_all;
stage = log.stage;
isDOE = stage == 0;
isAL = stage == 1;
isOnline = stage == 2;
scatter(X(isDOE,1), X(isDOE,2), 45, "r", "filled");
scatter(X(isAL,1), X(isAL,2), 45, "b", "filled");
if any(isOnline)
  scatter(X(isOnline,1), X(isOnline,2), 55, "g", "filled");
end

if isfield(spec, "trueG") && ~isempty(spec.trueG)
  x1 = linspace(spec.lb(1), spec.ub(1), 250);
  x2 = linspace(spec.lb(2), spec.ub(2), 250);
  [X1, X2] = meshgrid(x1, x2);
  G = spec.trueG([X1(:), X2(:)]);
  G = reshape(G, size(X1));
  contour(X1, X2, G, [0 0], "k", "LineWidth", 1.5);
end

grid on; box on;
xlabel("x1"); ylabel("x2");
title("采样点与真实边界");
legend({"pool","DOE","AL","Online","g(x)=0"}, "Location", "best");
saveas(f, fullfile(outDir, "samples_2d.png"));
close(f);
end

