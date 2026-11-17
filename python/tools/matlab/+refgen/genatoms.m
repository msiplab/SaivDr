function s = genatoms(c)
%GENATOMS Atomic images computed as in saivdr.dcnn.atomicimshow
%
%   The computation of atomicimshow without the montage display, in double
%   precision. atomicImages is patchsize(1) x patchsize(2) x nComponents x
%   nChsTotal, where the first atom is the DC atom of the last level,
%   followed by the AC atoms of levels nLevels, ..., 1.
args = namedargs2cell(c.params);
patchsize = c.patchsize;
lgraph = saivdr.dcnn.fcn_creatensoltlgraph2d([],'InputSize',patchsize,args{:});
dlnet = dlnetwork(lgraph);
L = dlnet.Learnables;
for iRow = 1:height(L)
    L.Value{iRow} = dlarray(pi*(2*rand(size(L.Value{iRow}))-1));
end
dlnet.Learnables = L;
nComponents = c.params.NumberOfComponents;
nChannels = c.params.NumberOfChannels;
decFactor = c.params.DecimationFactor;
nLevels = c.params.NumberOfLevels;
nChsPerLv = sum(nChannels);
nChsTotal = nLevels*(nChsPerLv-1)+1;
atomicImages = zeros([patchsize nComponents nChsTotal]);
dls = cell(nLevels+1,1);
for iRevLv = nLevels:-1:1
    if iRevLv == nLevels
        dls{nLevels+1} = dlarray(zeros([patchsize./(decFactor.^nLevels) nComponents]),'SSC');
        dls{nLevels} = dlarray(zeros([patchsize./(decFactor.^nLevels) ...
            nComponents*(nChsPerLv-1)]),'SSC');
    else
        dls{iRevLv} = dlarray(zeros([patchsize./(decFactor.^iRevLv) ...
            nComponents*(nChsPerLv-1)]),'SSC');
    end
end
idx = 1;
dld = dls;
dld{nLevels+1}(round(end/2),round(end/2),1:nComponents) = ones(1,1,nComponents);
atomicImages(:,:,1:nComponents,idx) = extractdata(dlnet.predict(dld{:}));
idx = idx+1;
for iRevLv = nLevels:-1:1
    for iAtom = 1:nChsPerLv-1
        dld = dls;
        for iCmp = 1:nComponents
            dld{iRevLv}(round(end/2),round(end/2),(iCmp-1)*(nChsPerLv-1)+iAtom) = 1;
        end
        atomicImages(:,:,1:nComponents,idx) = extractdata(dlnet.predict(dld{:}));
        idx = idx+1;
    end
end
s.meta_params = jsonencode(c.params);
s.meta_input_names = strjoin(dlnet.InputNames,',');
s.patchsize = patchsize;
s.atoms = atomicImages;
s.atoms_size = refgen.padsize(atomicImages,4);
s = refgen.collectparams(dlnet,s);
end
