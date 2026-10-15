function cases = layercases()
%LAYERCASES Test cases of the saivdr.dcnn layers
%
%   Spatial sizes and batch sizes are kept small so that the reference
%   data stay compact. Most cases are in double precision; a few are
%   repeated in single precision.
import refgen.mkcase
N = 2;
cases = {};
str = @(v) strrep(num2str(v),' ','');

%% Block DCT / IDCT 2-D
for dec = {[1 1],[2 2],[2 4],[4 1],[4 4]}
    d = dec{1};
    for nComps = [1 3]
        sz = [16 8];
        name = sprintf('BlockDct2d_dec%s_cmp%d',str(d),nComps);
        p = struct('DecimationFactor',d,'NumberOfComponents',nComps);
        outs = repmat({'coef2d'},1,nComps);
        cases{end+1,1} = (mkcase(name,'nsoltBlockDct2dLayer',p,{'image2d',[sz nComps N]},outs));
        name = sprintf('BlockIdct2d_dec%s_cmp%d',str(d),nComps);
        ins = repmat({'coef2d',[prod(d) sz./d N]},nComps,1);
        cases{end+1,1} = (mkcase(name,'nsoltBlockIdct2dLayer',p,ins,{'image2d'}));
    end
end
p = struct('DecimationFactor',[2 2],'NumberOfComponents',1);
cases{end+1,1} = (mkcase('BlockDct2d_dec22_cmp1','nsoltBlockDct2dLayer',p,...
    {'image2d',[8 8 1 N]},{'coef2d'},'DataType','single'));
cases{end+1,1} = (mkcase('BlockIdct2d_dec22_cmp1','nsoltBlockIdct2dLayer',p,...
    {'coef2d',[4 4 4 N]},{'image2d'},'DataType','single'));

%% Block DCT / IDCT 3-D
for dec = {[1 1 1],[2 2 2],[1 2 4],[4 4 2],[2 4 4]}
    d = dec{1};
    sz = [8 8 8];
    p = struct('DecimationFactor',d,'NumberOfComponents',1);
    cases{end+1,1} = (mkcase(sprintf('BlockDct3d_dec%s_cmp1',str(d)),'nsoltBlockDct3dLayer',p,...
        {'image3d',[sz 1 N]},{'coef3d'}));
    cases{end+1,1} = (mkcase(sprintf('BlockIdct3d_dec%s_cmp1',str(d)),'nsoltBlockIdct3dLayer',p,...
        {'coef3d',[prod(d) sz./d N]},{'image3d'}));
end
p = struct('DecimationFactor',[2 2 2],'NumberOfComponents',1);
cases{end+1,1} = (mkcase('BlockDct3d_dec222_cmp1','nsoltBlockDct3dLayer',p,...
    {'image3d',[8 8 8 1 N]},{'coef3d'},'DataType','single'));
cases{end+1,1} = (mkcase('BlockIdct3d_dec222_cmp1','nsoltBlockIdct3dLayer',p,...
    {'coef3d',[8 4 4 4 N]},{'image3d'},'DataType','single'));

%% Initial / final rotations 2-D (ps >= ceil(nDecs/2) and pa >= floor(nDecs/2))
for nch = {[1 1],[2 2],[3 3],[4 4]}
    c = nch{1};
    for dec = {[1 1],[1 2],[2 2],[2 4]}
        d = dec{1};
        if c(1) < ceil(prod(d)/2) || c(2) < floor(prod(d)/2)
            continue
        end
        for ndl = [false true]
            p = struct('NumberOfChannels',c,'DecimationFactor',d,'NoDcLeakage',ndl);
            tag = sprintf('nch%s_dec%s_ndl%d',str(c),str(d),ndl);
            cases{end+1,1} = (mkcase(['InitialRotation2d_' tag],'nsoltInitialRotation2dLayer',p,...
                {'coef2d',[prod(d) 4 4 N]},{'coef2d'},'Learnable',true,'RandomMus',true));
            cases{end+1,1} = (mkcase(['FinalRotation2d_' tag],'nsoltFinalRotation2dLayer',p,...
                {'coef2d',[sum(c) 4 4 N]},{'coef2d'},'Learnable',true,'RandomMus',true));
        end
    end
end
p = struct('NumberOfChannels',[4 4],'DecimationFactor',[2 2],'NoDcLeakage',true);
cases{end+1,1} = (mkcase('InitialRotation2d_nch44_dec22_ndl1','nsoltInitialRotation2dLayer',p,...
    {'coef2d',[4 4 4 N]},{'coef2d'},'Learnable',true,'RandomMus',true,'DataType','single'));
cases{end+1,1} = (mkcase('FinalRotation2d_nch44_dec22_ndl1','nsoltFinalRotation2dLayer',p,...
    {'coef2d',[8 4 4 N]},{'coef2d'},'Learnable',true,'RandomMus',true,'DataType','single'));

%% Initial / final rotations 3-D
for nch = {[4 4],[5 5]}
    c = nch{1};
    for dec = {[1 1 1],[2 2 2],[1 2 4]}
        d = dec{1};
        if c(1) < ceil(prod(d)/2) || c(2) < floor(prod(d)/2)
            continue
        end
        for ndl = [false true]
            p = struct('NumberOfChannels',c,'DecimationFactor',d,'NoDcLeakage',ndl);
            tag = sprintf('nch%s_dec%s_ndl%d',str(c),str(d),ndl);
            cases{end+1,1} = (mkcase(['InitialRotation3d_' tag],'nsoltInitialRotation3dLayer',p,...
                {'coef3d',[prod(d) 4 4 4 N]},{'coef3d'},'Learnable',true,'RandomMus',true));
            cases{end+1,1} = (mkcase(['FinalRotation3d_' tag],'nsoltFinalRotation3dLayer',p,...
                {'coef3d',[sum(c) 4 4 4 N]},{'coef3d'},'Learnable',true,'RandomMus',true));
        end
    end
end

%% Intermediate rotations
for nch = {[2 2],[3 3],[4 4]}
    c = nch{1};
    for mode = {'Analysis','Synthesis'}
        p = struct('NumberOfChannels',c,'Mode',mode{1});
        tag = sprintf('nch%s_%s',str(c),lower(mode{1}));
        cases{end+1,1} = (mkcase(['IntermediateRotation2d_' tag],'nsoltIntermediateRotation2dLayer',p,...
            {'coef2d',[sum(c) 4 4 N]},{'coef2d'},'Learnable',true,'RandomMus',true));
    end
end
for nch = {[4 4],[5 5]}
    c = nch{1};
    for mode = {'Analysis','Synthesis'}
        p = struct('NumberOfChannels',c,'Mode',mode{1});
        tag = sprintf('nch%s_%s',str(c),lower(mode{1}));
        cases{end+1,1} = (mkcase(['IntermediateRotation3d_' tag],'nsoltIntermediateRotation3dLayer',p,...
            {'coef3d',[sum(c) 4 4 4 N]},{'coef3d'},'Learnable',true,'RandomMus',true));
    end
end
p = struct('NumberOfChannels',[4 4],'Mode','Analysis');
cases{end+1,1} = (mkcase('IntermediateRotation2d_nch44_analysis','nsoltIntermediateRotation2dLayer',p,...
    {'coef2d',[8 4 4 N]},{'coef2d'},'Learnable',true,'RandomMus',true,'DataType','single'));

%% Atom extensions
for nch = {[3 3],[4 4]}
    c = nch{1};
    for dir = {'Right','Left','Down','Up'}
        for target = {'Sum','Difference'}
            p = struct('NumberOfChannels',c,'Direction',dir{1},'TargetChannels',target{1});
            tag = sprintf('nch%s_%s_%s',str(c),lower(dir{1}),lower(target{1}));
            cases{end+1,1} = (mkcase(['AtomExtension2d_' tag],'nsoltAtomExtension2dLayer',p,...
                {'coef2d',[sum(c) 4 6 N]},{'coef2d'}));
        end
    end
end
for dir = {'Right','Left','Down','Up','Back','Front'}
    for target = {'Sum','Difference'}
        p = struct('NumberOfChannels',[4 4],'Direction',dir{1},'TargetChannels',target{1});
        tag = sprintf('nch44_%s_%s',lower(dir{1}),lower(target{1}));
        cases{end+1,1} = (mkcase(['AtomExtension3d_' tag],'nsoltAtomExtension3dLayer',p,...
            {'coef3d',[8 4 6 5 N]},{'coef3d'}));
    end
end
p = struct('NumberOfChannels',[4 4],'Direction','Right','TargetChannels','Sum');
cases{end+1,1} = (mkcase('AtomExtension2d_nch44_right_sum','nsoltAtomExtension2dLayer',p,...
    {'coef2d',[8 4 6 N]},{'coef2d'},'DataType','single'));

%% Channel separation / concatenation, component separation, identity
p = struct();
cases{end+1,1} = (mkcase('ChannelSeparation2d_nch33','nsoltChannelSeparation2dLayer',p,...
    {'coef2d',[6 4 5 N]},{'image2d','image2d'}));
cases{end+1,1} = (mkcase('ChannelConcatenation2d_nch33','nsoltChannelConcatenation2dLayer',p,...
    {'image2d',[4 5 5 N];'image2d',[4 5 1 N]},{'coef2d'}));
cases{end+1,1} = (mkcase('ChannelSeparation3d_nch44','nsoltChannelSeparation3dLayer',p,...
    {'coef3d',[8 4 5 3 N]},{'image3d','image3d'}));
cases{end+1,1} = (mkcase('ChannelConcatenation3d_nch44','nsoltChannelConcatenation3dLayer',p,...
    {'image3d',[4 5 3 7 N];'image3d',[4 5 3 1 N]},{'coef3d'}));
for nComps = [1 3]
    p = struct('NumberOfComponents',nComps);
    cases{end+1,1} = (mkcase(sprintf('ComponentSeparation2d_cmp%d',nComps),...
        'nsoltComponentSeparation2dLayer',p,{'image2d',[4 5 2*nComps N]},...
        repmat({'image2d'},1,nComps),'Backward',false));
end
cases{end+1,1} = (mkcase('Identity','nsoltIdentityLayer',struct(),{'coef2d',[6 4 5 N]},{'coef2d'}));

%% Subband (de)serialization
for nLevels = 1:3
    for dec = {[2 2],[1 2]}
        d = dec{1};
        for nch = {[3 3],[4 4]}
            c = nch{1};
            orgdim = [16 16];
            p = struct('OriginalDimension',orgdim,'NumberOfChannels',c,...
                'DecimationFactor',d,'NumberOfLevels',nLevels);
            tag = sprintf('lv%d_dec%s_nch%s',nLevels,str(d),str(c));
            ins = cell(nLevels+1,2);
            for iLv = 1:nLevels
                ins(iLv,:) = {'image2d',[orgdim./d.^iLv sum(c)-1 N]};
            end
            ins(nLevels+1,:) = {'image2d',[orgdim./d.^nLevels 1 N]};
            nElements = sum(cellfun(@(s) prod(s(1:3)),ins(:,2)));
            cases{end+1,1} = (mkcase(['SubbandSerialization2d_' tag],...
                'nsoltSubbandSerialization2dLayer',p,ins,{'serial2d'}));
            cases{end+1,1} = (mkcase(['SubbandDeserialization2d_' tag],...
                'nsoltSubbandDeserialization2dLayer',p,{'serial2d',[nElements 1 1 N]},...
                repmat({'image2d'},1,nLevels+1)));
        end
    end
end
for nLevels = 1:2
    for dec = {[2 2 2],[1 2 2],[2 1 2],[2 2 1]}
        d = dec{1};
        c = [5 5];
        orgdim = [8 8 8];
        p = struct('OriginalDimension',orgdim,'NumberOfChannels',c,...
            'DecimationFactor',d,'NumberOfLevels',nLevels);
        tag = sprintf('lv%d_dec%s_nch%s',nLevels,str(d),str(c));
        ins = cell(nLevels+1,2);
        for iLv = 1:nLevels
            ins(iLv,:) = {'image3d',[orgdim./d.^iLv sum(c)-1 N]};
        end
        ins(nLevels+1,:) = {'image3d',[orgdim./d.^nLevels 1 N]};
        nElements = sum(cellfun(@(s) prod(s(1:4)),ins(:,2)));
        cases{end+1,1} = (mkcase(['SubbandSerialization3d_' tag],...
            'nsoltSubbandSerialization3dLayer',p,ins,{'serial3d'}));
        cases{end+1,1} = (mkcase(['SubbandDeserialization3d_' tag],...
            'nsoltSubbandDeserialization3dLayer',p,{'serial3d',[nElements 1 1 1 N]},...
            repmat({'image3d'},1,nLevels+1)));
    end
end
end
