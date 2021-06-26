function [analysislgraph,synthesislgraph] = fcn_attachserializationlayers(...
    analysislgraph,synthesislgraph,szPatchTrn)
%FCN_ATTACHSERIALIZATIONLAYERS
%
% Setting up the analysis dictionary (adjoint operator) by copying
% synthesis dictionary parameters to the analyisis dictionary
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

expinitlayer = '^Lv1_Cmp1+_V0$';
expdctlayer = '^Lv\d+_E0$';
nLayers = height(analysislgraph.Layers);
nLevels = 0;
for iLayer = 1:nLayers
    layer = analysislgraph.Layers(iLayer);
    layerName = layer.Name;
    if ~isempty(regexp(layerName,expdctlayer,'once'))
        nLevels = nLevels + 1;
    end
    if ~isempty(regexp(layerName,expinitlayer,'once'))
        nChannels = layer.NumberOfChannels;
        decFactor = layer.DecimationFactor;
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
    analysislgraph = analysislgraph.connectLayers(...
        [ strLv '_AcOut'], ['Sb_Srz/' strLv '_SbAcIn']);
    if iLv == nLevels
        analysislgraph = analysislgraph.connectLayers(...
            [ strLv '_DcOut'], ['Sb_Srz/' strLv '_SbDcIn']);
    end
end

%%

expfinallayer = '^Lv1_Cmp1+_V0~$';
expidctlayer = '^Lv\d+_E0~$';
nLayers = height(synthesislgraph.Layers);
nLevels = 0;
for iLayer = 1:nLayers
    layer = synthesislgraph.Layers(iLayer);
    layerName = layer.Name;
    if ~isempty(regexp(layerName,expidctlayer,'once'))
        nLevels = nLevels + 1;
    end
    if ~isempty(regexp(layerName,expfinallayer,'once'))
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


for iLv = 1:nLevels
    %synthesislgraph = synthesislgraph.replaceLayer(...
    %   ['Lv' num2str(iLv) '_Ac feature input'],... 
    %   nsoltIdentityLayer('Name',['Lv' num2str(iLv) '_AcIn']));
    synthesislgraph = synthesislgraph.removeLayers(...
       ['Lv' num2str(iLv) '_Ac feature input']);    
end
%synthesislgraph = synthesislgraph.replaceLayer(...
%    ['Lv' num2str(nLevels) '_Dc feature input'],...
%    nsoltIdentityLayer('Name',['Lv' num2str(nLevels) '_DcIn']));
synthesislgraph = synthesislgraph.removeLayers(...
    ['Lv' num2str(nLevels) '_Dc feature input']);

for iLv = 1:nLevels
    strLv = [ 'Lv' num2str(iLv) ];
    synthesislgraph = synthesislgraph.connectLayers(...
        ['Sb_Dsz/' strLv '_SbAcOut'], [ strLv '_AcIn']);
    if iLv == nLevels
        synthesislgraph = synthesislgraph.connectLayers(...
            ['Sb_Dsz/' strLv '_SbDcOut'],[ strLv '_DcIn']);
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

