function saveModelAndLog(model, log, outDir, stage)
%SAVEMODELANDLOG Save model(s) and log to .mat files.

arguments
  model (1,1) struct
  log (1,1) struct
  outDir (1,1) string
  stage (1,1) string = "offline"
end

if ~exist(outDir, "dir")
  mkdir(outDir);
end

save(fullfile(outDir, "log.mat"), "log");

switch stage
  case "offline"
    save(fullfile(outDir, "model_offline.mat"), "model");
  case "online"
    save(fullfile(outDir, "model_online.mat"), "model");
  otherwise
    save(fullfile(outDir, "model.mat"), "model");
end
end

