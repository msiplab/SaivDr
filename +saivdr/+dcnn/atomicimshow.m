function atomicimshow(synthesisnet,patchsize)
%FCN_ATOMICIMSHOW
%
% Display atomic images of NSOLT synthesis network
%
% Requirements: MATLAB R2020a
%
% Copyright (c) 2020, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%
import saivdr.dcnn.*

% Extraction of information
targetlayer = 'Lv1_V0~';
nLayers = length(synthesisnet.Layers);
nLevels = 0;
isSerialized = false;
for iLayer = 1:nLayers
    layer = synthesisnet.Layers(iLayer);
    if strcmp(layer.Name,'Sb_Dsz')
        isSerialized = true;
    end
    if strcmp(layer.Name,targetlayer)
        nChannels = layer.NumberOfChannels;
        decFactor = layer.DecimationFactor;
    end
    if ~isempty(strfind(layer.Name,'E0'))
        nLevels = nLevels + 1;
    end
end
nChsPerLv = sum(nChannels);
nChsTotal = nLevels*(nChsPerLv-1)+1;

% Patch Size
DIMENSION = 2;
MARGIN = 2;
if nargin < 2
    estPpOrder = floor([1 1]*sqrt(nLayers/(DIMENSION*nLevels)));
    estKernelExt = decFactor.*(estPpOrder+1);
    for iLv = 2:nLevels
        estKernelExt = (estKernelExt-1).*(decFactor+1)+1;
    end
    maxDecFactor = decFactor.^nLevels;
    
    patchsize = (ceil(estKernelExt./maxDecFactor)+MARGIN).*maxDecFactor;
end

% Remove deserialization
if isSerialized
    synthesislgraph = layerGraph(synthesisnet);
    synthesislgraph = synthesislgraph.removeLayers({'Sb_Dsz','Subband images'});
    [~,synthesislgraph] = fcn_replaceinputlayers([],...
        synthesislgraph,patchsize);
    synthesisnet = dlnetwork(synthesislgraph);
end

% Calculation of atomic images
atomicImages = zeros([patchsize 1 nChsTotal]);

% Impluse arrays
dls = cell(nLevels+1,1);
for iRevLv = nLevels:-1:1
    if iRevLv == nLevels
        dls{nLevels+1} = dlarray(...
            zeros([patchsize./(decFactor.^nLevels) 1],'single'),...
            'SSC');
        dls{nLevels} = dlarray(...
            zeros([patchsize./(decFactor.^nLevels) (nChsPerLv-1)],'single'),...
            'SSC');
    else
        dls{iRevLv} = dlarray(...
            zeros([patchsize./(decFactor.^iRevLv) (nChsPerLv-1)],'single'),...
            'SSC');
    end
end

% Impluse responses
idx = 1;
dld = dls;
dld{nLevels+1}(round(end/2),round(end/2),1)  = 1;
atomicImages(:,:,1,idx) = ...
    extractdata(synthesisnet.predict(dld{:}));
idx = idx+1;
for iRevLv = nLevels:-1:1
    for iAtom = 1:nChsPerLv-1
        dld = dls;
        dld{iRevLv}(round(end/2),round(end/2),iAtom)  = 1;
        atomicImages(:,:,1,idx) = ...
            extractdata(synthesisnet.predict(dld{:}));
        idx = idx+1;
    end
end

mRows = 2^(nextpow2(sqrt(nChsTotal))-1);
mCols = ceil(nChsTotal/mRows);
montage(imresize(atomicImages,8,'nearest')+.5,...
    'Size',[mRows mCols],'BorderSize',[2 2]);
end

