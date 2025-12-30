function outDir = makeOutDir(baseDir, prefix)
%MAKEOUTDIR Create a timestamped output directory.

arguments
  baseDir (1,1) string = "out"
  prefix (1,1) string = "run"
end

ts = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
outDir = fullfile(char(baseDir), char(ts + "_" + prefix));
if ~exist(outDir, "dir")
  mkdir(outDir);
end
end

