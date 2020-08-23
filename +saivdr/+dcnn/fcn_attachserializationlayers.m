function [analysislgraph,synthesislgraph] = fcn_attachserializationlayers(...
    analysislgraph,synthesislgraph,szPatchTrn)
%FCN_ATTACHSERIALIZATIONLAYERS
%
% Setting up the analysis dictionary (adjoint operator) by copying
% synthesis dictionary parameters to the analyisis dictionary
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

nLayers = length(analysislgraph.Layers);
nLevels = 0;
for iLayer = 1:nLayers
    layerName = analysislgraph.Layers(iLayer).Name;
    if contains(layerName,'_E0')
        nLevels = nLevels + 1;
    end
    if contains(layerName,'Lv1_V0')
        nChannels = analysislgraph.Layers(iLayer).NumberOfChannels;
        decFactor = analysislgraph.Layers(iLayer).DecimationFactor;
    end
end

%
sbSerializationLayer = nsoltSubbandSerialization2dLayer(...
    'Name','Sb_Srz',...
    'OriginalDimension',szPatchTrn,...
    'NumberOfLevels',nLevels,...
    'DecimationFactor',decFactor,...
    'NumberOfChannels',nChannels);
analysislgraph = analysislgraph.addLayers(...
    sbSerializationLayer);

for iLv = 1:nLevels
    strLv = [ 'Lv' num2str(iLv) ];
    if iLv < nLevels
        analysislgraph = analysislgraph.connectLayers(...
            [ strLv '_AcOut'], ['Sb_Srz/' strLv '_SbIn']);
    else
        analysislgraph = analysislgraph.connectLayers(...
            [ strLv '_DcAcOut'], ['Sb_Srz/' strLv '_SbIn']);
    end
end

%%
nLayers = length(synthesislgraph.Layers);
nLevels = 0;
for iLayer = 1:nLayers
    layerName = synthesislgraph.Layers(iLayer).Name;
    if contains(layerName,'_E0~')
        nLevels = nLevels + 1;
    end
    if contains(layerName,'Lv1_V0~')
        nChannels = synthesislgraph.Layers(iLayer).NumberOfChannels;
        decFactor = synthesislgraph.Layers(iLayer).DecimationFactor;
    end    
end

sbDeserializationLayer =  nsoltSubbandDeserialization2dLayer(...
    'Name','Sb_Dsz',...
    'OriginalDimension',szPatchTrn,...
    'NumberOfLevels',nLevels,...
    'DecimationFactor',decFactor,...
    'NumberOfChannels',nChannels);
synthesislgraph = synthesislgraph.addLayers(...
    sbDeserializationLayer);

for iLv = 1:nLevels-1
    synthesislgraph = synthesislgraph.replaceLayer(...
       ['Lv' num2str(iLv) ' subband images'],... 
       nsoltIdentityLayer('Name',['Lv' num2str(iLv) '_AcIn']));
end
synthesislgraph = synthesislgraph.replaceLayer(...
    ['Lv' num2str(nLevels) ' subband images'],...
    nsoltIdentityLayer('Name',['Lv' num2str(nLevels) '_DcAcIn']));

for iLv = 1:nLevels
    strLv = [ 'Lv' num2str(iLv) ];
    if iLv < nLevels
        synthesislgraph = synthesislgraph.connectLayers(...
            ['Sb_Dsz/' strLv '_SbOut'], [ strLv '_AcIn']);
    else
        synthesislgraph = synthesislgraph.connectLayers(...
            ['Sb_Dsz/' strLv '_SbOut'],[ strLv '_DcAcIn']);
    end
end

synthesislgraph = synthesislgraph.addLayers(...
    imageInputLayer(...
    sbDeserializationLayer.InputSize,...
    'Name','Subband images',...
    'Normalization','none'));
synthesislgraph = synthesislgraph.connectLayers(...
    'Subband images','Sb_Dsz');
end

