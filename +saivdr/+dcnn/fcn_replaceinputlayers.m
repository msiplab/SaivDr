function [analysislgraph, synthesislgraph] = fcn_replaceinputlayers(analysislgraph,synthesislgraph,imagesize)
%FCN_CREATENSOLTLGRAPHS2D
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

% Analysis layer graph
if ~isempty(analysislgraph)
    analysislgraph = analysislgraph.replaceLayer(...
        'Lv1_In',...
        imageInputLayer(imagesize,'Name','Input image','Normalization','none'));
end

% Synthesis layer graph
if ~isempty(synthesislgraph)
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
    
    for iLv = 1:nLevels-1
        synthesislgraph = synthesislgraph.replaceLayer(...
            ['Lv' num2str(iLv) '_AcIn'],...
            imageInputLayer([imagesize./(decFactor.^iLv) (sum(nChannels)-1)],...
            'Name',['Lv' num2str(iLv) ' subband images'],'Normalization','none'));
    end
    synthesislgraph = synthesislgraph.replaceLayer(...
        ['Lv' num2str(nLevels) '_DcAcIn'],...
        imageInputLayer([imagesize./(decFactor.^nLevels) sum(nChannels)],...
        'Name',['Lv' num2str(nLevels) ' subband images'],'Normalization','none'));
end
end