%MAIN_SEC_4_2_ATMIMG_NSOLT2 Display Atomic Images of 2-D NSOLT
%
% This script was used for observing atomic images of 2-D NSOLT
% in Section 4.2.
%
% The following materials are also reproduced:
%
%  * tiff/fig03x.tif  ( x in {a,b,c,...,h} )
%  * materials/fig03.tex
%
% SVN identifier:
% $Id: main_pkg_atmimg.m 875 2016-01-20 01:51:38Z sho $
%
% Requirements: MATLAB R2014a
%
%  * Signal Processing Toolbox
%  * Image Processing Toolbox
%
% Recommended:
%
%  * MATLAB Coder
%
% Copyright (c) 2014-2016, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
%
nfig = 3;       % Figure number

%% Preparation
close all
isExport = true;

%% Conditions
imgName = 'barbara';   % Name of the image used for dictionary learning
imgSize = 32;        % Size of the image used for dictionary learning
nChs  = [ 12 12 ];   % # of channels [ ps pa ]
nDecs = [ 4 4 ];     % Downsampling factor [ M0 M1 ]
nOrds = [ 2 2 ];     % Polyphase order [ N0 N1 ]
nCoefs = 128;        % # of Coefs. for dictionary learning
nLevels = 1;         % # of tree levels
vm = 1;              % # of vanishing moments
nIters = 20;         % # of iterations for dictionary learning
isBest = false;     % Extract the best results in terms of PSNR

%%
fileName = sprintf(...
    'results/nsolt_d%dx%d_c%d+%d_o%d+%d_v%d_l%d_n%d_%s_ufc0.mat',...
    nDecs(1),nDecs(2),nChs(1),nChs(2),...
    nOrds(1),nOrds(2),vm,nLevels,...
    nCoefs,[ imgName num2str(imgSize)]);
if exist(fileName,'file')
    display(fileName)
    S = load(fileName,'lppufbs','mses');
    lppufbs = S.lppufbs;
    mses_   = cell2mat(S.mses);
    if isBest
        [~,idx] = min(mses_(1:nIters));  %#ok
    elseif nIters < length(lppufbs)
        idx = nIters;
    else
        idx = length(lppufbs);
    end
    lppufb = lppufbs{idx};
    hf1 = figure;
    pos = get(hf1,'Position');
    prewidth = pos(3);
    height = pos(4);
    pstwidth = max(nChs)*height/2;
    prex = pos(1);
    pstx = prex - (pstwidth-prewidth)/2;
    pos(1) = pstx;
    pos(3) = pstwidth;
    set(hf1,'Position',pos);
    clf
    lppufb = saivdr.dictionary.utility.fcn_upgrade(lppufb);
    release(lppufb)
    set(lppufb,'OutputMode','SynthesisFilters');
    F = step(lppufb,[],[]);
    clf
    atmimshow(lppufb)
    if isExport
        for sidx=1:sum(nChs)
            id = ('a'+sidx-1);
            imwrite(imresize(F(:,:,sidx)+0.5,4*[size(F,1),size(F,2)],'nearest'),...
                sprintf('tiff/fig%02d%c.tif',nfig,id))
        end
    end
else
    fprintf('%s : no such file\n',fileName)
end

%% Generate LaTeX file
sw = StringWriter();

sw.addcr('%#! latex muramatsu')
sw.addcr('%')
sw.addcr('% $Id: main_pkg_atmimg.m 875 2016-01-20 01:51:38Z sho $');
sw.addcr('%')
sw.addcr('\begin{figure}[tb]')
sw.addcr('\centering')
for iCh = 1:sum(nChs)
    id = ('a'+iCh-1);
    sw.addcr('\\includegraphics[width=.07\\columnwidth]{fig%02d%c}',nfig,id)
end
sw.add(sprintf('\\caption{$%d \\times %d$ ',nDecs(1)*(nOrds(1)+1),nDecs(2)*(nOrds(2)+1)))
sw.add('pixel learned atomic images ')
sw.add(sprintf('for {\\it %s}',imgName))
sw.add(' with $\mathbf{M}=\text{diag}(M_0,M_1)=\text{diag}(')
sw.add(sprintf('%d,%d',nDecs(1),nDecs(2)))
sw.add(')$, $M=|\det(\mathbf{M})|=M_0\times M_1=')
sw.add(sprintf('%d\\times %d',nDecs(1),nDecs(2)))
sw.add('$, $P=p_{\rm s}+p_\mathrm{a}=')
sw.add(sprintf('%d+%d',nChs(1),nChs(2)))
sw.add('$, $\bar{\mathbf{n}}=(N_0,N_1)^T=(')
sw.add(sprintf('%d,%d',nOrds(1),nOrds(2)))
sw.addcr(')^T$.}')
sw.addcr(sprintf('\\label{fig:%02d}',nfig))
sw.addcr('\end{figure}')
sw.add('\endinput')

%%
disp(sw)
write(sw,sprintf('materials/fig%02d.tex',nfig))
