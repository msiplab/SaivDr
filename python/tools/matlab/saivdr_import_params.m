function net = saivdr_import_params(net,file,varargin)
%SAIVDR_IMPORT_PARAMS Import NSOLT parameters saved by saivdr.dcnn (PyTorch)
%
%   net = saivdr_import_params(net,file) sets the Angles, Mus and NoDcLeakage
%   of the rotation layers of NET (a dlnetwork or a layerGraph) from the
%   MAT-file FILE written by saivdr.dcnn.matlab_compat.save_params in Python
%   or by saivdr_export_nsoltx.
%
%   net = saivdr_import_params(net,file,'Strict',false) skips the layers in
%   FILE that NET does not have, e.g. the synthesis layers for an analyzer.
%
% Copyright (c) 2026, Shogo MURAMATSU
p = inputParser;
addParameter(p,'Strict',true)
parse(p,varargin{:})
s = load(file);
isDlnet = isa(net,'dlnetwork');
if isDlnet
    lgraph = layerGraph(net);
else
    lgraph = net;
end
names = cellfun(@char,s.names,'UniformOutput',false);
if ~p.Results.Strict
    keep = ismember(names,{lgraph.Layers.Name});
    names = names(keep);
    s.angles = s.angles(keep);
    s.mus = s.mus(keep);
    s.nodcleakage = s.nodcleakage(keep);
end
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
