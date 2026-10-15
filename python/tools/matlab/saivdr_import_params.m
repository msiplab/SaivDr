function net = saivdr_import_params(net,file)
%SAIVDR_IMPORT_PARAMS Import NSOLT parameters saved by saivdr.dcnn (PyTorch)
%
%   net = saivdr_import_params(net,file) sets the Angles, Mus and NoDcLeakage
%   of the rotation layers of NET (a dlnetwork or a layerGraph) from the
%   MAT-file FILE written by saivdr.dcnn.matlab_compat.save_params in Python.
%
% Copyright (c) 2026, Shogo MURAMATSU
s = load(file);
isDlnet = isa(net,'dlnetwork');
if isDlnet
    lgraph = layerGraph(net);
else
    lgraph = net;
end
names = cellfun(@char,s.names,'UniformOutput',false);
for i = 1:numel(names)
    layer = lgraph.Layers(strcmp({lgraph.Layers.Name},names{i}));
    if isempty(layer)
        error('Layer %s is not found',names{i})
    end
    if ~isnan(s.nodcleakage(i))
        layer.NoDcLeakage = logical(s.nodcleakage(i));
    end
    layer.Mus = double(s.mus{i}(:));
    layer.Angles = double(s.angles{i}(:));
    lgraph = lgraph.replaceLayer(names{i},layer);
end
if isDlnet
    net = dlnetwork(lgraph);
    % Keep the raw angles, which replaceLayer may have updated
    L = net.Learnables;
    for i = 1:numel(names)
        row = strcmp(L.Layer,names{i}) & strcmp(L.Parameter,'Angles');
        L.Value{row} = dlarray(double(s.angles{i}(:)));
    end
    net.Learnables = L;
else
    net = lgraph;
end
end
