function s = collectparams(dlnet,s,gW)
%COLLECTPARAMS Store the angles, mus and NoDcLeakage of the rotation layers
%
%   s.names, s.angles, s.mus are cell columns aligned with each other and
%   s.nodcleakage is NaN for the layers without the property. When the
%   gradient table gW is given, s.dldw holds the gradients of the angles.
L = dlnet.Learnables;
isAngles = strcmp(L.Parameter,'Angles');
names = L.Layer(isAngles);
n = numel(names);
s.names = cell(n,1);
s.angles = cell(n,1);
s.mus = cell(n,1);
s.nodcleakage = nan(n,1);
if nargin > 2
    s.dldw = cell(n,1);
end
idx = find(isAngles);
for i = 1:n
    layer = dlnet.Layers(strcmp({dlnet.Layers.Name},names{i}));
    s.names{i} = char(names{i});
    s.angles{i} = double(extractdata(L.Value{idx(i)}));
    s.mus{i} = double(layer.Mus(:));
    if isprop(layer,'NoDcLeakage')
        s.nodcleakage(i) = double(layer.NoDcLeakage);
    end
    if nargin > 2
        s.dldw{i} = double(extractdata(gW.Value{idx(i)}));
    end
end
end
