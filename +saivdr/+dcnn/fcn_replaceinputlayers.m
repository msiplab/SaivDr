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
analysislgraph = analysislgraph.replaceLayer(...
    'Lv1_In',...
    imageInputLayer(imagesize,'Name','Input image','Normalization','none'));
nChannels = analysislgraph.Layers(2).NumberOfChannels;
decFactors = analysislgraph.Layers(2).DecimationFactor;
nLevels = str2double(analysislgraph.Layers(end).Name(3));

% Synthesis layer graph
for iLv = 1:nLevels-1
    synthesislgraph = synthesislgraph.replaceLayer(...
        ['Lv' num2str(iLv) '_AcIn'],...
        imageInputLayer([imagesize./(decFactors.^iLv) (sum(nChannels)-1)],...
        'Name',['Lv' num2str(iLv) ' subband images'],'Normalization','none'));   
end
synthesislgraph = synthesislgraph.replaceLayer(...
    ['Lv' num2str(nLevels) '_DcAcIn'],...
    imageInputLayer([imagesize./(decFactors.^nLevels) sum(nChannels)],...
    'Name',['Lv' num2str(nLevels) ' subband images'],'Normalization','none'));
end