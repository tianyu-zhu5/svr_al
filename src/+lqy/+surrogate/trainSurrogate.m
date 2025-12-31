function surr = trainSurrogate(XTrain, gTrain, cfg, spec, normModel, hp)
%TRAINSURROGATE Train A-mean surrogate:
% - ghat(x) from a single main model (controllable)
% - sigma(x) from bootstrap ensemble (for uncertainty only)

arguments
  XTrain (:,:) double
  gTrain (:,1) double
  cfg (1,1) struct
  spec (1,1) struct %#ok<INUSA>
  normModel (1,1) struct
  hp (1,1) struct
end

w = lqy.svr.computeWeights(gTrain, cfg);
ZTrain = lqy.norm.zscoreApply(XTrain, normModel);

main = lqy.surrogate.trainMainModel(ZTrain, gTrain, w, cfg, hp);
ens = lqy.surrogate.trainSigmaBootstrap(ZTrain, gTrain, w, cfg, hp);

surr = struct();
surr.main = main;
surr.ens = ens;
surr.hp = hp;
surr.normModel = normModel;
surr.meta = struct("strategy","A-mean");
end

