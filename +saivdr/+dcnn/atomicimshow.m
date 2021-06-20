function atomicimshow(synthesisnet,patchsize,scale)
%FCN_ATOMICIMSHOW
%
% Display atomic images of NSOLT synthesis network
%
% Requirements: MATLAB R2020a
%
% Copyright (c) 2020-2021, Shogo MURAMATSU
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
if nargin < 3
    scale = 1;
end

% Extraction of information
expfinallayer = '^Lv1_Cmp1+_V0~$';
expidctlayer = '^Lv\d+_E0~$';
nLayers = length(synthesisnet.Layers);
nLevels = 0;
isSerialized = false;
for iLayer = 1:nLayers
    layer = synthesisnet.Layers(iLayer);
    if strcmp(layer.Name,'Sb_Dsz')
        isSerialized = true;
    end
    if ~isempty(regexp(layer.Name,expfinallayer,'once'))
        nChannels = layer.NumberOfChannels;
        decFactor = layer.DecimationFactor;
    end
    if ~isempty(regexp(layer.Name,expidctlayer,'once'))
        nLevels = nLevels + 1;
        if nLevels == 1
            nComponents = layer.NumInputs;
        end
    end
end
nChsPerLv = sum(nChannels);
nChsTotal = nLevels*(nChsPerLv-1)+1;

% Patch Size
DIMENSION = 2;
MARGIN = 2;
if nargin < 2 || isempty(patchsize)
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
    synthesislgraph = synthesislgraph.removeLayers( { 'Sb_Dsz', 'Subband images' });
    %
    for iLv = 1:nLevels
        synthesislgraph = synthesislgraph.addLayers(...            
            imageInputLayer(...
            [patchsize./(decFactor.^iLv) nComponents*(sum(nChannels)-1)],...
            'Name',['Lv' num2str(iLv) '_Ac feature input'],...
            'Normalization','none'));
        synthesislgraph = synthesislgraph.connectLayers(...
            ['Lv' num2str(iLv) '_Ac feature input'],...
            ['Lv' num2str(iLv) '_AcIn']);
    end
    %
    synthesislgraph = synthesislgraph.addLayers(...
        imageInputLayer(...
        [patchsize./(decFactor.^nLevels) nComponents],...
        'Name',['Lv' num2str(nLevels) '_Dc feature input'],...
        'Normalization','none'));
    synthesislgraph = synthesislgraph.connectLayers(...
        ['Lv' num2str(nLevels) '_Dc feature input'],...
        ['Lv' num2str(nLevels) '_DcIn']);
    %
    synthesisnet = dlnetwork(synthesislgraph);
end

% Calculation of atomic images
atomicImages = zeros([patchsize 1 nChsTotal],'single');

% Impluse arrays
dls = cell(nLevels+1,1);
for iRevLv = nLevels:-1:1
    if iRevLv == nLevels
        dls{nLevels+1} = dlarray(...
            zeros([patchsize./(decFactor.^nLevels) nComponents],'single'),...
            'SSC');
        dls{nLevels} = dlarray(...
            zeros([patchsize./(decFactor.^nLevels) nComponents*(nChsPerLv-1)],'single'),...
            'SSC');
    else
        dls{iRevLv} = dlarray(...
            zeros([patchsize./(decFactor.^iRevLv) nComponents*(nChsPerLv-1)],'single'),...
            'SSC');
    end
end

% Impluse responses
idx = 1;
dld = dls;
dld{nLevels+1}(round(end/2),round(end/2),1:nComponents)  = ones(1,1,nComponents);
atomicImages(:,:,1:nComponents,idx) = ...
    extractdata(synthesisnet.predict(dld{:}));
idx = idx+1;
for iRevLv = nLevels:-1:1
    for iAtom = 1:nChsPerLv-1
        dld = dls;
        for iCmp = 1:nComponents
            dld{iRevLv}(round(end/2),round(end/2),(iCmp-1)*(nChsPerLv-1)+iAtom)  = 1;
        end
        atomicImages(:,:,1:nComponents,idx) = ...
            extractdata(synthesisnet.predict(dld{:}));
        idx = idx+1;
    end
end

mRows = 2^(nextpow2(sqrt(nChsTotal))-1);
mCols = ceil(nChsTotal/mRows);
montage(imresize(scale*atomicImages,8,'nearest')+.5,...
    'Size',[mRows mCols],'BorderSize',[2 2]);
end

