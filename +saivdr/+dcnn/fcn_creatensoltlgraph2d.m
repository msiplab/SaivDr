function nsoltLgraph = ...
    fcn_creatensoltlgraph2d(varargin)
%FCN_CREATENSOLTLGRAPHS2D
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
p = inputParser;
addParameter(p,'InputSize',[32 32])
addParameter(p,'NumberOfComponents',1)
addParameter(p,'NumberOfChannels',[2 2])
addParameter(p,'DecimationFactor',[2 2])
addParameter(p,'PolyPhaseOrder',[0 0])
addParameter(p,'NumberOfLevels',1);
addParameter(p,'NumberOfVanishingMoments',[1 1]);
addParameter(p,'Mode','Whole');
parse(p,varargin{:})

% Layer constructor function goes here.
nComponents = p.Results.NumberOfComponents;
inputSize = [p.Results.InputSize nComponents];
nChannels = p.Results.NumberOfChannels;
decFactor = p.Results.DecimationFactor;
ppOrder = p.Results.PolyPhaseOrder;
nLevels = p.Results.NumberOfLevels;
noDcLeakage = p.Results.NumberOfVanishingMoments;
if isscalar(noDcLeakage)
    noDcLeakage = [1 1]*noDcLeakage; 
end
mode = p.Results.Mode;

if strcmp(mode,'Whole')
    isAnalyzer = true;
    isSynthesizer = true;
elseif strcmp(mode,'Analyzer')
    isAnalyzer = true;
    isSynthesizer = false;
elseif strcmp(mode,'Synthesizer')
    isAnalyzer = false;
    isSynthesizer = true;
else
    error('Mode should be in { ''Whole'', ''Analyzer'', ''Synthesizer'' }');
end

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

%%
blockDctLayers = cell(nLevels);
analysisLayers = cell(nLevels,nComponents);
blockIdctLayers = cell(nLevels);
synthesisLayers = cell(nLevels,nComponents);
for iLv = 1:nLevels
    strLv = sprintf('Lv%0d_',iLv);
    
    % Initial blocks
    blockDctLayers{iLv} = nsoltBlockDct2dLayer('Name',[strLv 'E0'],...
        'DecimationFactor',decFactor,...
        'NumberOfComponents',nComponents);
    for iCmp = 1:nComponents
        strCmp = sprintf('Cmp%0d_',iCmp);
        analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
            nsoltInitialRotation2dLayer('Name',[strLv strCmp 'V0'],...
            'NumberOfChannels',nChannels,'DecimationFactor',decFactor,...
            'NoDcLeakage',noDcLeakage(1))
            ];
    end
    % Final blocks
    blockIdctLayers{iLv} = nsoltBlockIdct2dLayer('Name',[strLv 'E0~'],...
        'DecimationFactor',decFactor,...
        'NumberOfComponents',nComponents);
    for iCmp = 1:nComponents
        strCmp = sprintf('Cmp%0d_',iCmp);
        synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
            nsoltFinalRotation2dLayer('Name',[strLv strCmp 'V0~'],...
            'NumberOfChannels',nChannels,'DecimationFactor',decFactor,...
            'NoDcLeakage',noDcLeakage(2))
            ];
    end
    
    for iCmp = 1:nComponents
        strCmp = sprintf('Cmp%0d_',iCmp);
        % Atom extension in horizontal
        for iOrderH = 2:2:ppOrder(2)
            analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                nsoltAtomExtension2dLayer('Name',[strLv strCmp 'Qh' num2str(iOrderH-1) 'rd'],...
                'NumberOfChannels',nChannels,'Direction','Right','TargetChannels','Difference')
                nsoltIntermediateRotation2dLayer('Name',[strLv strCmp 'Vh' num2str(iOrderH-1)],...
                'NumberOfChannels',nChannels,'Mode','Analysis','Mus',-1)
                nsoltAtomExtension2dLayer('Name',[strLv strCmp 'Qh' num2str(iOrderH) 'ls'],...
                'NumberOfChannels',nChannels,'Direction','Left','TargetChannels','Sum')
                nsoltIntermediateRotation2dLayer('Name',[strLv strCmp 'Vh' num2str(iOrderH) ],...
                'NumberOfChannels',nChannels,'Mode','Analysis')
                ];
            synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                nsoltAtomExtension2dLayer('Name',[strLv strCmp 'Qh' num2str(iOrderH-1) 'rd~'],...
                'NumberOfChannels',nChannels,'Direction','Left','TargetChannels','Difference')
                nsoltIntermediateRotation2dLayer('Name',[strLv strCmp 'Vh' num2str(iOrderH-1) '~'],...
                'NumberOfChannels',nChannels,'Mode','Synthesis','Mus',-1)
                nsoltAtomExtension2dLayer('Name',[strLv strCmp 'Qh' num2str(iOrderH) 'ls~'],...
                'NumberOfChannels',nChannels,'Direction','Right','TargetChannels','Sum')
                nsoltIntermediateRotation2dLayer('Name',[strLv strCmp 'Vh' num2str(iOrderH) '~'],...
                'NumberOfChannels',nChannels,'Mode','Synthesis')
                ];
        end
        % Atom extension in vertical
        for iOrderV = 2:2:ppOrder(1)
            analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                nsoltAtomExtension2dLayer('Name',[strLv strCmp 'Qv' num2str(iOrderV-1) 'dd'],...
                'NumberOfChannels',nChannels,'Direction','Down','TargetChannels','Difference')
                nsoltIntermediateRotation2dLayer('Name',[strLv strCmp 'Vv' num2str(iOrderV-1)],...
                'NumberOfChannels',nChannels,'Mode','Analysis','Mus',-1)
                nsoltAtomExtension2dLayer('Name',[strLv strCmp 'Qv' num2str(iOrderV) 'us'],...
                'NumberOfChannels',nChannels,'Direction','Up','TargetChannels','Sum')
                nsoltIntermediateRotation2dLayer('Name',[strLv strCmp 'Vv' num2str(iOrderV)],...
                'NumberOfChannels',nChannels,'Mode','Analysis')
                ];
            synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                nsoltAtomExtension2dLayer('Name',[strLv strCmp 'Qv' num2str(iOrderV-1) 'dd~'],...
                'NumberOfChannels',nChannels,'Direction','Up','TargetChannels','Difference')
                nsoltIntermediateRotation2dLayer('Name',[strLv strCmp 'Vv' num2str(iOrderV-1) '~'],...
                'NumberOfChannels',nChannels,'Mode','Synthesis','Mus',-1)
                nsoltAtomExtension2dLayer('Name',[strLv strCmp 'Qv' num2str(iOrderV) 'us~'],...
                'NumberOfChannels',nChannels,'Direction','Down','TargetChannels','Sum')
                nsoltIntermediateRotation2dLayer('Name',[strLv strCmp 'Vv' num2str(iOrderV) '~'],...
                'NumberOfChannels',nChannels,'Mode','Synthesis')
                ];
        end
        
        % Channel separation and concatenation
        analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
            nsoltChannelSeparation2dLayer('Name',[strLv strCmp 'Sp'])
            ];
        synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
            nsoltChannelConcatenation2dLayer('Name',[strLv strCmp 'Cn'])
            ];
    end
    
end

%%
nsoltLgraph = layerGraph;

%% Analysis layers
if isAnalyzer
    % Level 1
    iLv = 1;
    strLv = sprintf('Lv%0d_',iLv);
    nsoltLgraph = nsoltLgraph.addLayers(...
        [ imageInputLayer(inputSize,...
        'Name','Image input',...
        'Normalization','none'),...
        nsoltIdentityLayer(...
        'Name',[strLv 'In']),...
        blockDctLayers{iLv}
    ]);
    for iCmp = 1:nComponents
        strCmp = sprintf('Cmp%0d_',iCmp);
        nsoltLgraph = nsoltLgraph.addLayers(analysisLayers{iLv,iCmp});
        if nComponents > 1
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [strLv 'E0/out' num2str(iCmp)], [strLv strCmp 'V0']);
        else
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [strLv 'E0'], [strLv strCmp 'V0']);
        end
    end
    % Output
    if nComponents > 1
        nsoltLgraph = nsoltLgraph.addLayers(...
            depthConcatenationLayer(nComponents,'Name',[strLv 'AcOut']));        
        nsoltLgraph = nsoltLgraph.addLayers(...
            depthConcatenationLayer(nComponents,'Name',[strLv 'DcOut']));
    else
        nsoltLgraph = nsoltLgraph.addLayers(...
            nsoltIdentityLayer('Name',[strLv 'AcOut']));
        nsoltLgraph = nsoltLgraph.addLayers(...
            nsoltIdentityLayer('Name',[strLv 'DcOut']));
    end
    if nComponents > 1
        for iCmp = 1:nComponents
            strCmp = sprintf('Cmp%0d_',iCmp);
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [strLv strCmp 'Sp/ac'], [strLv 'AcOut/in' num2str(iCmp)]);
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [strLv strCmp 'Sp/dc'], [strLv 'DcOut/in' num2str(iCmp)]);
        end
    else
        strCmp = 'Cmp1_';
        nsoltLgraph = nsoltLgraph.connectLayers(...
            [strLv strCmp 'Sp/ac'], [strLv 'AcOut/in' ]);
        nsoltLgraph = nsoltLgraph.connectLayers(...
            [strLv strCmp 'Sp/dc'], [strLv 'DcOut/in' ]);
    end
    % Level n > 1
    for iLv = 2:nLevels
        strLv = sprintf('Lv%0d_',iLv);
        strLvPre = sprintf('Lv%0d_',iLv-1);
        nsoltLgraph = nsoltLgraph.addLayers([
            nsoltIdentityLayer('Name',[strLv 'In']),...
            blockDctLayers{iLv}]);
        nsoltLgraph = nsoltLgraph.connectLayers([strLvPre 'DcOut'],[strLv 'In']);
        for iCmp = 1:nComponents
            strCmp = sprintf('Cmp%0d_',iCmp);
            nsoltLgraph = nsoltLgraph.addLayers(analysisLayers{iLv,iCmp});
            if nComponents > 1
                nsoltLgraph = nsoltLgraph.connectLayers(...
                    [strLv 'E0/out' num2str(iCmp)], [strLv strCmp 'V0']);
            else
                nsoltLgraph = nsoltLgraph.connectLayers(...
                    [strLv 'E0'], [strLv strCmp 'V0']);
            end
        end
        % Output
        if nComponents > 1
            nsoltLgraph = nsoltLgraph.addLayers(...
                depthConcatenationLayer(nComponents,'Name',[strLv 'AcOut']));            
            nsoltLgraph = nsoltLgraph.addLayers(...
                depthConcatenationLayer(nComponents,'Name',[strLv 'DcOut']));
        else
            nsoltLgraph = nsoltLgraph.addLayers(...
                nsoltIdentityLayer('Name',[strLv 'AcOut']));
            nsoltLgraph = nsoltLgraph.addLayers(...
                nsoltIdentityLayer('Name',[strLv 'DcOut']));
        end
        if nComponents > 1
            for iCmp = 1:nComponents
                strCmp = sprintf('Cmp%0d_',iCmp);
                nsoltLgraph = nsoltLgraph.connectLayers(...
                    [strLv strCmp 'Sp/ac'], [strLv 'AcOut/in' num2str(iCmp)]);
                nsoltLgraph = nsoltLgraph.connectLayers(...
                    [strLv strCmp 'Sp/dc'], [strLv 'DcOut/in' num2str(iCmp)]);
            end
        else
            strCmp = 'Cmp1_';
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [strLv strCmp 'Sp/ac'], [strLv 'AcOut/in' ]);
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [strLv strCmp 'Sp/dc'], [strLv 'DcOut/in' ]);
        end
    end
end
%{
if ~isSynthesizer
    for iLv = 1:nLevels
        strLv = sprintf('Lv%0d_',iLv);
        nsoltLgraph = nsoltLgraph.addLayers(...
            regressionLayer('Name',[ strLv 'Ac feature output']));
        nsoltLgraph = nsoltLgraph.connectLayers(...
            [ strLv 'AcOut'],[ strLv 'Ac feature output']);
    end
    strLv = sprintf('Lv%0d_',nLevels);
    nsoltLgraph = nsoltLgraph.addLayers(...
        regressionLayer('Name',[ strLv 'Dc feature output']));
    nsoltLgraph = nsoltLgraph.connectLayers(...
            [ strLv 'DcOut'],[ strLv 'Dc feature output']);    
end
%}


%% Synthesis layers
if isSynthesizer
    % Level N
    iLv = nLevels;
    strLv = sprintf('Lv%0d_',iLv);
    nsoltLgraph = nsoltLgraph.addLayers(...
        nsoltComponentSeparation2dLayer(nComponents,'Name',[strLv 'DcIn']));
    nsoltLgraph = nsoltLgraph.addLayers(...
        nsoltComponentSeparation2dLayer(nComponents,'Name',[strLv 'AcIn']));
    if nComponents > 1
        for iCmp = 1:nComponents
            strCmp = sprintf('Cmp%0d_',iCmp);
            nsoltLgraph = nsoltLgraph.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [strLv 'AcIn/out' num2str(iCmp) ], [strLv strCmp 'Cn/ac']);
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [strLv 'DcIn/out' num2str(iCmp) ], [strLv strCmp 'Cn/dc']);
        end
    else
        strCmp = 'Cmp1_';
        nsoltLgraph = nsoltLgraph.addLayers(synthesisLayers{iLv,1}(end:-1:1));
        nsoltLgraph = nsoltLgraph.connectLayers(...
            [strLv 'AcIn/out' ], [strLv strCmp 'Cn/ac']);
        nsoltLgraph = nsoltLgraph.connectLayers(...
            [strLv 'DcIn/out' ], [strLv strCmp 'Cn/dc']);
    end
    nsoltLgraph = nsoltLgraph.addLayers([
        blockIdctLayers{iLv},...
        nsoltIdentityLayer('Name',[strLv 'Out'])
        ]);
    for iCmp = 1:nComponents
        strCmp = sprintf('Cmp%0d_',iCmp);
        if nComponents > 1
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [strLv strCmp 'V0~'],[strLv 'E0~/in' num2str(iCmp)]);
        else
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [strLv strCmp 'V0~'],[strLv 'E0~']);
        end
    end
    
    % Level n < N
    for iLv = nLevels-1:-1:1
        strLv = sprintf('Lv%0d_',iLv);
        strLvPre = sprintf('Lv%0d_',iLv+1);
        if nComponents > 1
            nsoltLgraph = nsoltLgraph.addLayers(...
                nsoltComponentSeparation2dLayer(nComponents,'Name',[strLv 'DcIn']));
            nsoltLgraph = nsoltLgraph.addLayers(...
                nsoltComponentSeparation2dLayer(nComponents,'Name',[strLv 'AcIn']));            
        else
            nsoltLgraph = nsoltLgraph.addLayers(...
                nsoltIdentityLayer('Name',[strLv 'DcIn']));
            nsoltLgraph = nsoltLgraph.addLayers(...
                nsoltIdentityLayer('Name',[strLv 'AcIn']));            
        end
        nsoltLgraph = nsoltLgraph.connectLayers([strLvPre 'Out'],[strLv 'DcIn']);
        if nComponents > 1
            for iCmp = 1:nComponents
                strCmp = sprintf('Cmp%0d_',iCmp);
                nsoltLgraph = nsoltLgraph.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
                nsoltLgraph = nsoltLgraph.connectLayers(...
                    [strLv 'AcIn/out' num2str(iCmp) ], [strLv strCmp 'Cn/ac']);
                nsoltLgraph = nsoltLgraph.connectLayers(...
                    [strLv 'DcIn/out' num2str(iCmp) ], [strLv strCmp 'Cn/dc']);
            end
        else
            strCmp = 'Cmp1_';
            nsoltLgraph = nsoltLgraph.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [strLv 'AcIn/out' ], [strLv strCmp 'Cn/ac']);
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [strLv 'DcIn/out'  ], [strLv strCmp 'Cn/dc']);
        end
        nsoltLgraph = nsoltLgraph.addLayers([
            blockIdctLayers{iLv},...
            nsoltIdentityLayer('Name',[strLv 'Out'])
            ]);
        for iCmp = 1:nComponents
            strCmp = sprintf('Cmp%0d_',iCmp);
            if nComponents > 1
                nsoltLgraph = nsoltLgraph.connectLayers(...
                    [strLv strCmp 'V0~'],[strLv 'E0~/in' num2str(iCmp)]);
            else
                nsoltLgraph = nsoltLgraph.connectLayers(...
                    [strLv strCmp 'V0~'], [strLv 'E0~']);
            end
        end
    end
    
    % Level 1
    %{
    nsoltLgraph = nsoltLgraph.addLayers(...
        regressionLayer('Name','Image output'));
    nsoltLgraph = nsoltLgraph.connectLayers('Lv1_Out','Image output');
    %}
end
if ~isAnalyzer
    for iLv = 1:nLevels
        strLv = sprintf('Lv%0d_',iLv);
        inputSubSize(1:2) = inputSize(1:2)./(decFactor.^iLv);
        inputSubSize(3) = nComponents*(sum(nChannels)-1); 
        nsoltLgraph = nsoltLgraph.addLayers(...
            imageInputLayer(inputSubSize,...
                'Name',[ strLv 'Ac feature input'],...
                'Normalization','none'));
            nsoltLgraph = nsoltLgraph.connectLayers(...
                [ strLv 'Ac feature input'],[ strLv 'AcIn'] );
    end
    strLv = sprintf('Lv%0d_',nLevels);
    inputSubSize(1:2) = inputSize(1:2)./(decFactor.^nLevels);
    inputSubSize(3) = nComponents;
    nsoltLgraph = nsoltLgraph.addLayers(...
        imageInputLayer(inputSubSize,...
        'Name',[ strLv 'Dc feature input'],...
        'Normalization','none'));
    if iLv == nLevels
        nsoltLgraph = nsoltLgraph.connectLayers(...
            [ strLv 'Dc feature input'],[ strLv 'DcIn']);
    end
end

%% Connect analyzer and synthesizer
if isAnalyzer && isSynthesizer
    strLv = sprintf('Lv%0d_',nLevels);
    nsoltLgraph = nsoltLgraph.connectLayers([strLv 'DcOut'],[strLv 'DcIn']);
    for iLv = nLevels:-1:1
        strLv = sprintf('Lv%0d_',iLv);
        nsoltLgraph = nsoltLgraph.connectLayers(...
            [strLv 'AcOut'],[strLv 'AcIn']);
    end
    nsoltLgraph = nsoltLgraph.addLayers(...
        regressionLayer('Name','Image output'));
    nsoltLgraph = nsoltLgraph.connectLayers('Lv1_Out','Image output');
end
end
