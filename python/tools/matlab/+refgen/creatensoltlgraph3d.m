function nsoltLgraph = creatensoltlgraph3d(nsoltLgraph,varargin)
%CREATENSOLTLGRAPH3D Reference 3-D NSOLT layer graph for the PyTorch port
%
%   The 3-D counterpart of saivdr.dcnn.fcn_creatensoltlgraph2d for a single
%   component, built only from the existing 3-D layers of saivdr.dcnn. The
%   atoms are extended in the depth, horizontal and vertical directions in
%   this order, as in saivdr.dictionary.nsoltx.NsoltAtomExtender3d. It is
%   the specification of saivdr.dcnn.create_nsolt_3d in Python.
%
% Copyright (c) 2026, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
import saivdr.dcnn.*
if isempty(nsoltLgraph)
    nsoltLgraph = layerGraph;
end
p = inputParser;
addParameter(p,'InputSize',[16 16 16])
addParameter(p,'NumberOfComponents',1)
addParameter(p,'NumberOfChannels',[4 4])
addParameter(p,'DecimationFactor',[2 2 2])
addParameter(p,'PolyPhaseOrder',[0 0 0])
addParameter(p,'NumberOfLevels',1);
addParameter(p,'NumberOfVanishingMoments',[1 1]);
addParameter(p,'Mode','Whole');
parse(p,varargin{:})
if p.Results.NumberOfComponents ~= 1
    error('Only a single component is supported')
end
inputSize = [p.Results.InputSize 1];
nChannels = p.Results.NumberOfChannels;
decFactor = p.Results.DecimationFactor;
ppOrder = p.Results.PolyPhaseOrder;
nLevels = p.Results.NumberOfLevels;
noDcLeakage = p.Results.NumberOfVanishingMoments;
if isscalar(noDcLeakage)
    noDcLeakage = [1 1]*noDcLeakage;
end
mode = p.Results.Mode;
isAnalyzer = any(strcmp(mode,{'Whole','Analyzer'}));
isSynthesizer = any(strcmp(mode,{'Whole','Synthesizer'}));
if nChannels(1) ~= nChannels(2) || any(mod(ppOrder,2))
    error('Only Type-I NSOLT with even polyphase orders is supported')
end

% (polyphase-order index, label, analysis dirs, synthesis dirs, suffixes)
steps = {
    3, 'd', {'Back','Front'}, {'Front','Back'}, {'bd','fs'}
    2, 'h', {'Right','Left'}, {'Left','Right'}, {'rd','ls'}
    1, 'v', {'Down','Up'},    {'Up','Down'},    {'dd','us'}
    };
for iLv = 1:nLevels
    strLv = sprintf('Lv%0d_',iLv);
    s = [strLv 'Cmp1_'];
    ana = nsoltInitialRotation3dLayer('Name',[s 'V0'],...
        'NumberOfChannels',nChannels,'DecimationFactor',decFactor,...
        'NoDcLeakage',noDcLeakage(1));
    syn = nsoltFinalRotation3dLayer('Name',[s 'V0~'],...
        'NumberOfChannels',nChannels,'DecimationFactor',decFactor,...
        'NoDcLeakage',noDcLeakage(2));
    for iStep = 1:size(steps,1)
        [idx,lbl,adirs,sdirs,sfx] = steps{iStep,:};
        for iOrder = 2:2:ppOrder(idx)
            o1 = num2str(iOrder-1);
            o2 = num2str(iOrder);
            ana = [ ana
                nsoltAtomExtension3dLayer('Name',[s 'Q' lbl o1 sfx{1}],...
                'NumberOfChannels',nChannels,'Direction',adirs{1},'TargetChannels','Difference')
                nsoltIntermediateRotation3dLayer('Name',[s 'V' lbl o1],...
                'NumberOfChannels',nChannels,'Mode','Analysis','Mus',-1)
                nsoltAtomExtension3dLayer('Name',[s 'Q' lbl o2 sfx{2}],...
                'NumberOfChannels',nChannels,'Direction',adirs{2},'TargetChannels','Sum')
                nsoltIntermediateRotation3dLayer('Name',[s 'V' lbl o2],...
                'NumberOfChannels',nChannels,'Mode','Analysis','Mus',-1)
                ]; %#ok<AGROW>
            syn = [ syn
                nsoltAtomExtension3dLayer('Name',[s 'Q' lbl o1 sfx{1} '~'],...
                'NumberOfChannels',nChannels,'Direction',sdirs{1},'TargetChannels','Difference')
                nsoltIntermediateRotation3dLayer('Name',[s 'V' lbl o1 '~'],...
                'NumberOfChannels',nChannels,'Mode','Synthesis','Mus',-1)
                nsoltAtomExtension3dLayer('Name',[s 'Q' lbl o2 sfx{2} '~'],...
                'NumberOfChannels',nChannels,'Direction',sdirs{2},'TargetChannels','Sum')
                nsoltIntermediateRotation3dLayer('Name',[s 'V' lbl o2 '~'],...
                'NumberOfChannels',nChannels,'Mode','Synthesis','Mus',-1)
                ]; %#ok<AGROW>
        end
    end
    analysisLayers{iLv} = [
        nsoltIdentityLayer('Name',[strLv 'In'])
        nsoltBlockDct3dLayer('Name',[strLv 'E0'],'DecimationFactor',decFactor)
        ana
        nsoltChannelSeparation3dLayer('Name',[s 'Sp'])
        ]; %#ok<AGROW>
    synthesisLayers{iLv} = [
        nsoltChannelConcatenation3dLayer('Name',[s 'Cn'])
        syn(end:-1:1)
        nsoltBlockIdct3dLayer('Name',[strLv 'E0~'],'DecimationFactor',decFactor)
        nsoltIdentityLayer('Name',[strLv 'Out'])
        ]; %#ok<AGROW>
end

if isAnalyzer
    nsoltLgraph = nsoltLgraph.addLayers(image3dInputLayer(inputSize,...
        'Name','Image input','Normalization','none'));
    for iLv = 1:nLevels
        strLv = sprintf('Lv%0d_',iLv);
        s = [strLv 'Cmp1_'];
        nsoltLgraph = nsoltLgraph.addLayers(analysisLayers{iLv});
        nsoltLgraph = nsoltLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'AcOut']));
        nsoltLgraph = nsoltLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'DcOut']));
        nsoltLgraph = nsoltLgraph.connectLayers([s 'Sp/ac'],[strLv 'AcOut']);
        nsoltLgraph = nsoltLgraph.connectLayers([s 'Sp/dc'],[strLv 'DcOut']);
        if iLv == 1
            nsoltLgraph = nsoltLgraph.connectLayers('Image input',[strLv 'In']);
        else
            nsoltLgraph = nsoltLgraph.connectLayers(sprintf('Lv%0d_DcOut',iLv-1),[strLv 'In']);
        end
    end
end
if isSynthesizer
    for iLv = nLevels:-1:1
        strLv = sprintf('Lv%0d_',iLv);
        s = [strLv 'Cmp1_'];
        nsoltLgraph = nsoltLgraph.addLayers(synthesisLayers{iLv});
        nsoltLgraph = nsoltLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'AcIn']));
        nsoltLgraph = nsoltLgraph.addLayers(nsoltIdentityLayer('Name',[strLv 'DcIn']));
        nsoltLgraph = nsoltLgraph.connectLayers([strLv 'AcIn'],[s 'Cn/ac']);
        nsoltLgraph = nsoltLgraph.connectLayers([strLv 'DcIn'],[s 'Cn/dc']);
        if iLv < nLevels
            nsoltLgraph = nsoltLgraph.connectLayers(sprintf('Lv%0d_Out',iLv+1),[strLv 'DcIn']);
        end
    end
end
if ~isAnalyzer
    for iLv = 1:nLevels
        strLv = sprintf('Lv%0d_',iLv);
        nsoltLgraph = nsoltLgraph.addLayers(image3dInputLayer(...
            [inputSize(1:3)./(decFactor.^iLv) sum(nChannels)-1],...
            'Name',[strLv 'Ac feature input'],'Normalization','none'));
        nsoltLgraph = nsoltLgraph.connectLayers([strLv 'Ac feature input'],[strLv 'AcIn']);
    end
    strLv = sprintf('Lv%0d_',nLevels);
    nsoltLgraph = nsoltLgraph.addLayers(image3dInputLayer(...
        [inputSize(1:3)./(decFactor.^nLevels) 1],...
        'Name',[strLv 'Dc feature input'],'Normalization','none'));
    nsoltLgraph = nsoltLgraph.connectLayers([strLv 'Dc feature input'],[strLv 'DcIn']);
end
if isAnalyzer && isSynthesizer
    strLv = sprintf('Lv%0d_',nLevels);
    nsoltLgraph = nsoltLgraph.connectLayers([strLv 'DcOut'],[strLv 'DcIn']);
    for iLv = nLevels:-1:1
        strLv = sprintf('Lv%0d_',iLv);
        nsoltLgraph = nsoltLgraph.connectLayers([strLv 'AcOut'],[strLv 'AcIn']);
    end
end
end
