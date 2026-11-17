function s = genlayer(c)
%GENLAYER Generate the reference data of a layer test case
args = namedargs2cell(c.params);
if strcmp(c.layer,'nsoltComponentSeparation2dLayer')
    layer = saivdr.dcnn.nsoltComponentSeparation2dLayer(c.params.NumberOfComponents);
else
    layer = feval(['saivdr.dcnn.' c.layer],args{:});
end
if c.learnable
    if c.randmus
        mus = sign(randn(numel(layer.Mus),1));
        mus(mus==0) = 1;
        layer.Mus = mus;
    end
    % Raw angles; with NoDcLeakage, some of them are regarded as zero
    layer.Angles = pi*(2*rand(numel(layer.Angles),1)-1);
end
nIn = size(c.inputs,1);
X = cell(1,nIn);
for iIn = 1:nIn
    X{iIn} = randn(c.inputs{iIn,2},c.dtype);
end
nOut = numel(c.outkinds);
Z = cell(1,nOut);
[Z{:}] = layer.predict(X{:});
s.meta_layer = c.layer;
s.meta_params = jsonencode(c.params);
s.meta_dtype = c.dtype;
s.meta_in_kinds = strjoin(c.inputs(:,1).',',');
s.meta_out_kinds = strjoin(c.outkinds,',');
s.meta_backward = double(c.backward);
for iIn = 1:nIn
    s.(sprintf('x%d',iIn)) = X{iIn};
    s.(sprintf('x%d_size',iIn)) = refgen.padsize(X{iIn},refgen.kindndims(c.inputs{iIn,1}));
end
for iOut = 1:nOut
    s.(sprintf('z%d',iOut)) = Z{iOut};
    s.(sprintf('z%d_size',iOut)) = refgen.padsize(Z{iOut},refgen.kindndims(c.outkinds{iOut}));
end
if c.learnable
    s.angles = layer.Angles;
    s.mus = layer.Mus;
    if isprop(layer,'NoDcLeakage')
        s.nodcleakage = double(layer.NoDcLeakage);
    end
end
if c.backward
    dLdZ = cellfun(@(z) randn(size(z),'like',z),Z,'UniformOutput',false);
    G = cell(1,nIn+c.learnable);
    [G{:}] = layer.backward(X{:},Z{:},dLdZ{:},[]);
    for iOut = 1:nOut
        s.(sprintf('dldz%d',iOut)) = dLdZ{iOut};
    end
    for iIn = 1:nIn
        s.(sprintf('dldx%d',iIn)) = G{iIn};
    end
    if c.learnable
        s.dldw = G{end};
    end
end
end
