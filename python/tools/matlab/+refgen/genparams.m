function s = genparams(c)
%GENPARAMS Export the parameters of a random network with saivdr_export_params
%
%   The file holds the exported parameters together with the input and the
%   outputs of the network. The import function is checked by a round trip.
s = refgen.gennetwork(c);
args = namedargs2cell(c.params);
lgraph = saivdr.dcnn.fcn_creatensoltlgraph2d([],'InputSize',c.inputSize,args{:});
dlnet = dlnetwork(lgraph);
tmp = [tempname '.mat'];
save(tmp,'-struct','s','names','angles','mus','nodcleakage','-v7');
dlnet = saivdr_import_params(dlnet,tmp);
exported = [tempname '.mat'];
saivdr_export_params(dlnet,exported);
t = load(exported);
delete(tmp); delete(exported);
for i = 1:numel(s.names)
    j = find(strcmp(t.names,s.names{i}));
    assert(isequal(t.angles{j},s.angles{i}) && isequal(t.mus{j},s.mus{i}),...
        'Round trip of %s failed',s.names{i})
end
X = dlarray(s.x1,'SSCB');
Z = cell(1,numel(dlnet.OutputNames));
[Z{:}] = predict(dlnet,X);
for i = 1:numel(Z)
    assert(max(abs(extractdata(Z{i})-s.(sprintf('z%d',i))),[],'all') < 1e-12,...
        'Imported network differs')
end
s = rmfield(s,intersect(fieldnames(s),{'dldw'}));
end
