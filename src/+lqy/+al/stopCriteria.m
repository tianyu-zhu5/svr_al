function [stop, reason] = stopCriteria(state, cfg)
%STOPCRITERIA Decide whether to stop based on budget and optional stability criteria.

arguments
  state (1,1) struct
  cfg (1,1) struct
end

stop = false;
reason = "continue";

if isfield(cfg, "budgetTotal") && state.Neval >= cfg.budgetTotal
  stop = true;
  reason = "budget";
  return;
end

if isfield(cfg, "stop") && isfield(cfg.stop, "useUminStable") && cfg.stop.useUminStable
  K = 3;
  delta = 1e-3;
  minIters = 10;
  if isfield(cfg.stop, "K"); K = cfg.stop.K; end
  if isfield(cfg.stop, "UminDelta"); delta = cfg.stop.UminDelta; end
  if isfield(cfg.stop, "minIters"); minIters = cfg.stop.minIters; end
  if numel(state.UminHist) < minIters
    return;
  end
  if numel(state.UminHist) >= K + 1
    recent = state.UminHist(end-K:end);
    if max(abs(diff(recent))) < delta
      stop = true;
      reason = "Umin_stable";
      return;
    end
  end
end

if isfield(cfg, "stop") && isfield(cfg.stop, "usePfConv") && cfg.stop.usePfConv
  K = 3;
  eta = 1e-2;
  minIters = 10;
  if isfield(cfg.stop, "K"); K = cfg.stop.K; end
  if isfield(cfg.stop, "PfEta"); eta = cfg.stop.PfEta; end
  if isfield(cfg.stop, "minIters"); minIters = cfg.stop.minIters; end
  if numel(state.PfHist) < minIters
    return;
  end
  if numel(state.PfHist) >= K + 1
    recent = state.PfHist(end-K:end);
    rel = abs(diff(recent)) ./ max(recent(1:end-1), 1e-6);
    if all(rel < eta)
      stop = true;
      reason = "Pf_converged";
      return;
    end
  end
end
end
