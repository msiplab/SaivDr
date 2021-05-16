function [analysisNsoltLgraph,synthesisNsoltLgraph] = ...
    fcn_creatensoltlgraphs2d(varargin)
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
p = inputParser;
addParameter(p,'NumberOfChannels',[2 2])
addParameter(p,'DecimationFactor',[2 2])
addParameter(p,'PolyPhaseOrder',[0 0])
addParameter(p,'NumberOfLevels',1);
addParameter(p,'NumberOfVanishingMoments',1);
parse(p,varargin{:})

% Layer constructor function goes here.
nChannels = p.Results.NumberOfChannels;
decFactor = p.Results.DecimationFactor;
ppOrder = p.Results.PolyPhaseOrder;
nLevels = p.Results.NumberOfLevels;
noDcLeakage = p.Results.NumberOfVanishingMoments;

if nChannels(1) ~= nChannels(2)
    throw(MException('NsoltLayer:InvalidNumberOfChannels',...
        '[%d %d] : Currently, Type-I NSOLT is only suported, where the even and odd channel numbers should be the same.',...
        nChannels(1),nChannels(2)))
end
if any(mod(ppOrder,2))
    throw(MException('NsoltLayer:InvalidPolyPhaseOrder',...
        '%d + %d : Currently, even polyphase orders are only supported.',...
        ppOrder(1),ppOrder(2)))
end

for iLv = 1:nLevels
    strLv = sprintf('Lv%0d_',iLv);
    
    % Initial blocks
    analysisLayers = [
        nsoltBlockDct2dLayer('Name',[strLv 'E0'],...
        'DecimationFactor',decFactor)
        nsoltInitialRotation2dLayer('Name',[strLv 'V0'],...
        'NumberOfChannels',nChannels,'DecimationFactor',decFactor,...
        'NoDcLeakage',noDcLeakage)
        ];
    synthesisLayers = [
        nsoltBlockIdct2dLayer('Name',[strLv 'E0~'],...
        'DecimationFactor',decFactor)
        nsoltFinalRotation2dLayer('Name',[strLv 'V0~'],...
        'NumberOfChannels',nChannels,'DecimationFactor',decFactor,...
        'NoDcLeakage',noDcLeakage)
        ];
    
    % Atom extension in horizontal
    for iOrderH = 2:2:ppOrder(2)
        analysisLayers = [ analysisLayers
            nsoltAtomExtension2dLayer('Name',[strLv 'Qh' num2str(iOrderH-1) 'rd'],...
            'NumberOfChannels',nChannels,'Direction','Right','TargetChannels','Difference')
            nsoltIntermediateRotation2dLayer('Name',[strLv 'Vh' num2str(iOrderH-1)],...
            'NumberOfChannels',nChannels,'Mode','Analysis','Mus',-1)
            nsoltAtomExtension2dLayer('Name',[strLv 'Qh' num2str(iOrderH) 'ls'],...
            'NumberOfChannels',nChannels,'Direction','Left','TargetChannels','Sum')
            nsoltIntermediateRotation2dLayer('Name',[strLv 'Vh' num2str(iOrderH) ],...
            'NumberOfChannels',nChannels,'Mode','Analysis')
            ];
        synthesisLayers = [ synthesisLayers
            nsoltAtomExtension2dLayer('Name',[strLv 'Qh' num2str(iOrderH-1) 'rd~'],...
            'NumberOfChannels',nChannels,'Direction','Left','TargetChannels','Difference')
            nsoltIntermediateRotation2dLayer('Name',[strLv 'Vh' num2str(iOrderH-1) '~'],...
            'NumberOfChannels',nChannels,'Mode','Synthesis','Mus',-1)
            nsoltAtomExtension2dLayer('Name',[strLv 'Qh' num2str(iOrderH) 'ls~'],...
            'NumberOfChannels',nChannels,'Direction','Right','TargetChannels','Sum')
            nsoltIntermediateRotation2dLayer('Name',[strLv 'Vh' num2str(iOrderH) '~'],...
            'NumberOfChannels',nChannels,'Mode','Synthesis')
            ];
    end
    % Atom extension in vertical
    for iOrderV = 2:2:ppOrder(1)
        analysisLayers = [ analysisLayers
            nsoltAtomExtension2dLayer('Name',[strLv 'Qv' num2str(iOrderV-1) 'dd'],...
            'NumberOfChannels',nChannels,'Direction','Down','TargetChannels','Difference')
            nsoltIntermediateRotation2dLayer('Name',[strLv 'Vv' num2str(iOrderV-1)],...
            'NumberOfChannels',nChannels,'Mode','Analysis','Mus',-1)
            nsoltAtomExtension2dLayer('Name',[strLv 'Qv' num2str(iOrderV) 'us'],...
            'NumberOfChannels',nChannels,'Direction','Up','TargetChannels','Sum')
            nsoltIntermediateRotation2dLayer('Name',[strLv 'Vv' num2str(iOrderV)],...
            'NumberOfChannels',nChannels,'Mode','Analysis')
            ];
        synthesisLayers = [ synthesisLayers
            nsoltAtomExtension2dLayer('Name',[strLv 'Qv' num2str(iOrderV-1) 'dd~'],...
            'NumberOfChannels',nChannels,'Direction','Up','TargetChannels','Difference')
            nsoltIntermediateRotation2dLayer('Name',[strLv 'Vv' num2str(iOrderV-1) '~'],...
            'NumberOfChannels',nChannels,'Mode','Synthesis','Mus',-1)
            nsoltAtomExtension2dLayer('Name',[strLv 'Qv' num2str(iOrderV) 'us~'],...
            'NumberOfChannels',nChannels,'Direction','Down','TargetChannels','Sum')
            nsoltIntermediateRotation2dLayer('Name',[strLv 'Vv' num2str(iOrderV) '~'],...
            'NumberOfChannels',nChannels,'Mode','Synthesis')
            ];
    end
    % Channel separation and concatenation
    analysisLayers = [ analysisLayers
        nsoltChannelSeparation2dLayer('Name',[strLv 'Sp'])
        ];
    synthesisLayers = [ synthesisLayers
        nsoltChannelConcatenation2dLayer('Name',[strLv 'Cn'])
        ];
    if iLv == 1
        analysisNsoltLgraph = layerGraph(analysisLayers);
        synthesisNsoltLgraph = layerGraph(synthesisLayers(end:-1:1));
    else
        analysisNsoltLgraph = analysisNsoltLgraph.addLayers(analysisLayers);
        synthesisNsoltLgraph = synthesisNsoltLgraph.addLayers(synthesisLayers(end:-1:1));
    end
    if iLv > 1
        strLvPre = sprintf('Lv%0d_',iLv-1);
        analysisNsoltLgraph = analysisNsoltLgraph.addLayers(nsoltIdentityLayer('Name',[strLvPre 'AcOut']));
        analysisNsoltLgraph = analysisNsoltLgraph.connectLayers([strLvPre 'Sp/ac'],[strLvPre 'AcOut']);
        analysisNsoltLgraph = analysisNsoltLgraph.connectLayers([strLvPre 'Sp/dc'],[strLv 'E0'] );
        %
        synthesisNsoltLgraph = synthesisNsoltLgraph.connectLayers([strLv 'E0~'],[strLvPre 'Cn/dc']);
        synthesisNsoltLgraph = synthesisNsoltLgraph.addLayers(nsoltIdentityLayer('Name',[strLvPre 'AcIn']));
        synthesisNsoltLgraph = synthesisNsoltLgraph.connectLayers([strLvPre 'AcIn'],[strLvPre 'Cn/ac']);
    end
end
analysisNsoltLgraph = analysisNsoltLgraph.addLayers(nsoltIdentityLayer('Name','Lv1_In'));
analysisNsoltLgraph = analysisNsoltLgraph.connectLayers('Lv1_In','Lv1_E0');
%
analysisNsoltLgraph = analysisNsoltLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'AcOut']));
analysisNsoltLgraph = analysisNsoltLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'DcOut']));
analysisNsoltLgraph = analysisNsoltLgraph.connectLayers([strLv 'Sp/ac'],[strLv 'AcOut']);    
analysisNsoltLgraph = analysisNsoltLgraph.connectLayers([strLv 'Sp/dc'],[strLv 'DcOut']);
%
synthesisNsoltLgraph = synthesisNsoltLgraph.addLayers(nsoltIdentityLayer('Name','Lv1_Out'));
synthesisNsoltLgraph = synthesisNsoltLgraph.connectLayers('Lv1_E0~','Lv1_Out');
%
synthesisNsoltLgraph = synthesisNsoltLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'AcIn']));
synthesisNsoltLgraph = synthesisNsoltLgraph.connectLayers([strLv 'AcIn'],[strLv 'Cn/ac']);
synthesisNsoltLgraph = synthesisNsoltLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'DcIn']));
synthesisNsoltLgraph = synthesisNsoltLgraph.connectLayers([strLv 'DcIn'],[strLv 'Cn/dc']);

end
