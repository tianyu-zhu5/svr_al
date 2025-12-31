function ghat = predictMainModel(main, ZQuery)
%PREDICTMAINMODEL Predict g-hat from the main surrogate model.

arguments
  main
  ZQuery (:,:) double
end

if isstruct(main) && isfield(main, "type") && main.type == "krr"
  ghat = lqy.svr.predictKRR(main, ZQuery);
else
  ghat = predict(main, ZQuery);
end
end

