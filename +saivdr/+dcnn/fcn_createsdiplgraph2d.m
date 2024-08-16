function sdipLgraph = fcn_createsdiplgraph2d(varargin)
%FCN_CREATESDIPLGRAPHS2D
%
% Requirements: MATLAB R2021a
%
% Copyright (c) 2021, Shogo MURAMATSU
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
addParameter(p,'NumberOfVanishingMoments',[1 1]);
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

analysisLayers = cell(nLevels);
synthesisLayers = cell(nLevels);
for iLv = 1:nLevels
    strLv = sprintf('Lv%0d_',iLv);
    
    % Initial blocks
    analysisLayers{iLv} = [
        nsoltBlockDct2dLayer('Name',[strLv 'E0'],...
        'DecimationFactor',decFactor)
        nsoltInitialRotation2dLayer('Name',[strLv 'V0'],...
        'NumberOfChannels',nChannels,'DecimationFactor',decFactor,...
        'NoDcLeakage',noDcLeakage(1))
        ];
    synthesisLayers{iLv} = [
        nsoltBlockIdct2dLayer('Name',[strLv 'E0~'],...
        'DecimationFactor',decFactor)
        nsoltFinalRotation2dLayer('Name',[strLv 'V0~'],...
        'NumberOfChannels',nChannels,'DecimationFactor',decFactor,...
        'NoDcLeakage',noDcLeakage(2))
        ];
    
    % Atom extension in horizontal
    for iOrderH = 2:2:ppOrder(2)
        analysisLayers{iLv} = [ analysisLayers{iLv}
            nsoltAtomExtension2dLayer('Name',[strLv 'Qh' num2str(iOrderH-1) 'rd'],...
            'NumberOfChannels',nChannels,'Direction','Right','TargetChannels','Difference')
            nsoltIntermediateRotation2dLayer('Name',[strLv 'Vh' num2str(iOrderH-1)],...
            'NumberOfChannels',nChannels,'Mode','Analysis','Mus',-1)
            nsoltAtomExtension2dLayer('Name',[strLv 'Qh' num2str(iOrderH) 'ls'],...
            'NumberOfChannels',nChannels,'Direction','Left','TargetChannels','Sum')
            nsoltIntermediateRotation2dLayer('Name',[strLv 'Vh' num2str(iOrderH) ],...
            'NumberOfChannels',nChannels,'Mode','Analysis')
            ];
        synthesisLayers{iLv} = [ synthesisLayers{iLv}
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
        analysisLayers{iLv} = [ analysisLayers{iLv}
            nsoltAtomExtension2dLayer('Name',[strLv 'Qv' num2str(iOrderV-1) 'dd'],...
            'NumberOfChannels',nChannels,'Direction','Down','TargetChannels','Difference')
            nsoltIntermediateRotation2dLayer('Name',[strLv 'Vv' num2str(iOrderV-1)],...
            'NumberOfChannels',nChannels,'Mode','Analysis','Mus',-1)
            nsoltAtomExtension2dLayer('Name',[strLv 'Qv' num2str(iOrderV) 'us'],...
            'NumberOfChannels',nChannels,'Direction','Up','TargetChannels','Sum')
            nsoltIntermediateRotation2dLayer('Name',[strLv 'Vv' num2str(iOrderV)],...
            'NumberOfChannels',nChannels,'Mode','Analysis')
            ];
        synthesisLayers{iLv} = [ synthesisLayers{iLv}
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
    analysisLayers{iLv} = [ analysisLayers{iLv}
        nsoltChannelSeparation2dLayer('Name',[strLv 'Sp'])
        ];
    synthesisLayers{iLv} = [ synthesisLayers{iLv}
        nsoltChannelConcatenation2dLayer('Name',[strLv 'Cn'])
        ];
end

% Analysis layers
strLv = 'Lv1_';
sdipLgraph = layerGraph(analysisLayers{1});
sdipLgraph = sdipLgraph.addLayers(nsoltIdentityLayer('Name','Lv1_In'));
sdipLgraph = sdipLgraph.connectLayers('Lv1_In','Lv1_E0');
for iLv = 2:nLevels
    strLv = sprintf('Lv%0d_',iLv);    
    strLvPre = sprintf('Lv%0d_',iLv-1);    
    sdipLgraph = sdipLgraph.addLayers(analysisLayers{iLv});
    sdipLgraph = sdipLgraph.addLayers(nsoltIdentityLayer('Name',[strLvPre 'AcOut']));
    sdipLgraph = sdipLgraph.addLayers(nsoltIdentityLayer('Name',[strLvPre 'DcOut']));
    sdipLgraph = sdipLgraph.connectLayers([strLvPre 'Sp/ac'],[strLvPre 'AcOut']);
    sdipLgraph = sdipLgraph.connectLayers([strLvPre 'Sp/dc'],[strLvPre 'DcOut'] );
    sdipLgraph = sdipLgraph.connectLayers([strLvPre 'DcOut'],[strLv 'E0'] );
end
sdipLgraph = sdipLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'AcOut']));
sdipLgraph = sdipLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'DcOut']));
sdipLgraph = sdipLgraph.connectLayers([strLv 'Sp/ac'],[strLv 'AcOut']);    
sdipLgraph = sdipLgraph.connectLayers([strLv 'Sp/dc'],[strLv 'DcOut']);

% Synthesis layers
strLv = sprintf('Lv%0d_',nLevels);        
sdipLgraph = sdipLgraph.addLayers(synthesisLayers{nLevels}(end:-1:1));
sdipLgraph = sdipLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'DcIn']));
sdipLgraph = sdipLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'AcIn']));
sdipLgraph = sdipLgraph.connectLayers([strLv 'DcIn'],[strLv 'Cn/dc']);
sdipLgraph = sdipLgraph.connectLayers([strLv 'AcIn'],[strLv 'Cn/ac']);
for iLv = nLevels-1:-1:1
    strLv = sprintf('Lv%0d_',iLv);   
    strLvPre = sprintf('Lv%0d_',iLv+1);
    sdipLgraph = sdipLgraph.addLayers(synthesisLayers{iLv}(end:-1:1));
    sdipLgraph = sdipLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'DcIn']));        
    sdipLgraph = sdipLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'AcIn']));
    sdipLgraph = sdipLgraph.connectLayers([strLvPre 'E0~'],[strLv 'DcIn']);
    sdipLgraph = sdipLgraph.connectLayers([strLv 'DcIn'],[strLv 'Cn/dc']);        
    sdipLgraph = sdipLgraph.connectLayers([strLv 'AcIn'],[strLv 'Cn/ac']);
end
%
sdipLgraph = sdipLgraph.addLayers(nsoltIdentityLayer('Name','Lv1_Out'));
sdipLgraph = sdipLgraph.connectLayers('Lv1_E0~','Lv1_Out');

% Connect analyzer and synthesizer
strLv = sprintf('Lv%0d_',nLevels); 
sdipLgraph = sdipLgraph.connectLayers([strLv 'DcOut'],[strLv 'DcIn']);
for iLv = nLevels:-1:1
    strLv = sprintf('Lv%0d_',iLv); 
        sdipLgraph = sdipLgraph.connectLayers([strLv 'AcOut'],[strLv 'AcIn']); 
end

end
