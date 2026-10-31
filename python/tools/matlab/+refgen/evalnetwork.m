function s = evalnetwork(dlnet,c)
%EVALNETWORK Evaluate a dlnetwork and collect the reference data
fmt = repmat('S',1,c.ndims);
inNames = dlnet.InputNames;
nIn = numel(inNames);
X = cell(1,nIn);
for iIn = 1:nIn
    layer = dlnet.Layers(strcmp({dlnet.Layers.Name},inNames{iIn}));
    X{iIn} = dlarray(randn([layer.InputSize c.batchSize],c.dtype),[fmt 'CB']);
end
nOut = numel(dlnet.OutputNames);
% Fix the upstream gradients before dlfeval
Z0 = cell(1,nOut);
[Z0{:}] = predict(dlnet,X{:});
dLdZ = cellfun(@(z) randn(size(z),'like',extractdata(z)),Z0,'UniformOutput',false);
[Z,gX,gW] = dlfeval(@lossgrad_,dlnet,X,dLdZ);

s.meta_params = jsonencode(c.params);
s.meta_ndims = c.ndims;
s.meta_dtype = c.dtype;
s.meta_input_names = strjoin(inNames,',');
s.meta_output_names = strjoin(dlnet.OutputNames,',');
kind = sprintf('image%dd',c.ndims);
for iIn = 1:nIn
    x = extractdata(X{iIn});
    s.(sprintf('x%d',iIn)) = x;
    s.(sprintf('x%d_size',iIn)) = refgen.padsize(x,refgen.kindndims(kind));
    s.(sprintf('dldx%d',iIn)) = extractdata(gX{iIn});
end
for iOut = 1:nOut
    z = extractdata(Z{iOut});
    s.(sprintf('z%d',iOut)) = z;
    s.(sprintf('z%d_size',iOut)) = refgen.padsize(z,refgen.kindndims(kind));
    s.(sprintf('dldz%d',iOut)) = dLdZ{iOut};
end
s = refgen.collectparams(dlnet,s,gW);
end

function [Z,gX,gW] = lossgrad_(dlnet,X,dLdZ)
nOut = numel(dlnet.OutputNames);
Z = cell(1,nOut);
[Z{:}] = forward(dlnet,X{:});
loss = 0;
for iOut = 1:nOut
    loss = loss + sum(Z{iOut}.*dLdZ{iOut},'all');
end
gX = cell(1,numel(X));
[gW,gX{:}] = dlgradient(loss,dlnet.Learnables,X{:});
end
