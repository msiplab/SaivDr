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
warning('Deprecated')

% Analysis layer graph
if ~isempty(analysislgraph)
    analysislgraph = analysislgraph.replaceLayer(...
        'Lv1_In',...
        imageInputLayer(imagesize,'Name','Input image','Normalization','none'));
end

% Synthesis layer graph
expfinallayer = '^Lv1_Cmp1+_V0~$';
expidctlayer = '^Lv\d+_E0~$';
if ~isempty(synthesislgraph)
    nLayers = length(synthesislgraph.Layers);
    nLevels = 0;
    for iLayer = 1:nLayers
        layer = synthesislgraph.Layers(iLayer);
        layerName = layer.Name;
        if ~isempty(regexp(layerName,expidctlayer,'once'))
            nLevels = nLevels + 1;
            if nLevels == 1
                nComponents  = layer.NumInputs;
            end
        end
        if ~isempty(regexp(layerName,expfinallayer,'once'))
            nChannels = layer.NumberOfChannels;
            decFactor = layer.DecimationFactor;
        end
    end
    
    for iLv = 1:nLevels
        synthesislgraph = synthesislgraph.replaceLayer(...
            ['Lv' num2str(iLv) '_AcIn'],...
            imageInputLayer([imagesize./(decFactor.^iLv) nComponents*(sum(nChannels)-1)],...
            'Name',['Lv' num2str(iLv) '_Ac feature input'],'Normalization','none'));
    end
    synthesislgraph = synthesislgraph.replaceLayer(...
        ['Lv' num2str(nLevels) '_DcIn'],...
        imageInputLayer([imagesize./(decFactor.^nLevels) nComponents],...
        'Name',['Lv' num2str(nLevels) '_Dc feature input'],'Normalization','none'));
end
end