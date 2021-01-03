%MAIN_SEC_4_2_TABLE3 Sparse Approximation with 2-D NSOLT
%
% This script was used for summarising the sparse approximation 
% performances in Section 4.2.
% 
% The following materials are also reproduced:
%
%  * tiff/fig02x.tif (x in {a,b,c,d})
%  * materials/fig02.tex
%  * materials/tab03.tex
%
% SVN identifier:
% $Id: main_pkg_psnrs.m 875 2016-01-20 01:51:38Z sho $
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
clear params
close all

nfig   = 2;     % Figure number
ntab   = 3;     % Table number
isEdit = false; % Edit mode

%% General parameters
params.isMonitoring = true;
imgSizeApx = 512;

%% Parameter settings for KSVDS
support.fcn_setup_ksvdsbox
params.Tdict        = 3;   % Atom sparsity
params.Tdata        = 2;   % Sparsity for learning
params.TdataImpl    = 2;   % Sparsity for approximation
params.blocksize    = 4;   % Block size
params.odctsize     = 5;   % ceil(params.blocksize*sqrt(redundancy));
params.useParallel  = true;

%% Parameter settings for NSOLT
params.chs          = [ 12 12 ]; % # of channels [ ps pa ]
params.dec          = [ 4 4 ];   % Downsampling factor [ M0 M1 ]
params.ord          = [ 2 2 ];   % Polyphase order [ N0 N1 ] 
params.nCoefs       = 32*32/8;   % # of Coefs. for learning
params.nCoefsImpl   = imgSizeApx^2/8; % # of Coefs. for approximation
params.index        = 'end';     % or positive integer or 'best'
params.isFixedCoefs = true;  
params.nUnfixedInitSteps = 0;
params.nVm          = 1;         % # of vanishing moments, No-DC-leakage option
params.nLevels      = 1;
params.filterDomain = 'frequency'; % or 'lattice_termination';

%% Condition setting 
imgSetLrnSksvd = { 'goldhill512', 'lena512', 'barbara512', 'baboon512' }; 
imgSetLrnNsolt = { 'goldhill32', 'lena32', 'barbara32', 'baboon32' }; 
imgSetApx ={ ...
    ['goldhill' num2str(imgSizeApx)], ...
    ['lena' num2str(imgSizeApx)], ...
    ['barbara' num2str(imgSizeApx)], ...
    ['baboon' num2str(imgSizeApx)] };
imgNames = { 'goldhill', 'lena', 'barbara', 'baboon' }; 
lvSet    = { 1, 2 }; % ƒcƒŠ[ƒŒƒxƒ‹

%% Pre-build of MEX files
if ~strcmp(params.filterDomain,'frequency')
    import saivdr.dictionary.nsoltx.mexsrcs.*
    chs = params.chs;
    fcn_autobuild_atomcnc2d([chs(1) chs(2)]);
    fcn_autobuild_atomext2d([chs(1) chs(2)]);
    if chs(1) == chs(2)
        fcn_autobuild_bb_type1(chs(1));
    else
        fcn_autobuild_bb_type2(chs(1),chs(2));
    end
end

%%
nLvSet    = length(lvSet);
nImgs     = length(imgNames);
psnrs_    = zeros(nLvSet+1,nImgs);
psnrtable = cell(nLvSet+1,nImgs);
for iImg = 1:nImgs
    % ompksvds
    if isEdit
        psnrs_(1,iImg) = iImg; % dummy
    else
        params.imgNameLrn = imgSetLrnSksvd{iImg};
        params.imgNameApx = imgSetApx{iImg};
        [ psnrs_(1,iImg), apxImg ] = support.fcn_ompsksvds2(params);
        imwrite(apxImg,sprintf('tiff/apx_%s_ksvds2.tif',imgNames{iImg}))
    end
    psnrtable{1,iImg} = num2str(psnrs_(1,iImg),'%6.2f');
    % ihtnsolt
    paramsTmp = cell(1,length(lvSet));
    for iLv = 1:nLvSet
        paramsTmp{iLv} = params;
        paramsTmp{iLv}.nLevels = lvSet{iLv};
        idx = iLv+1;
        if isEdit
            psnrs_(idx,iImg) = mod(10*(idx-1)+iImg,100); % dummy
        else
            params.imgNameLrn  = imgSetLrnNsolt{iImg};
            params.imgNameApx  = imgSetApx{iImg};
            params.nLevelsImpl = lvSet{iLv};
            [ psnrs_(idx,iImg), apxImg ] = ...
                support.fcn_ihtnsolt2(params);
            imwrite(apxImg,sprintf('tiff/apx_%s_nsolt2_lv%d.tif',...
                imgNames{iImg},params.nLevelsImpl))
        end
        psnrtable{idx,iImg} = num2str(psnrs_(idx,iImg),'%6.2f');
    end
end
%
[~,maxidx] = max(psnrs_);
for iImg = 1:nImgs
    psnrtable{maxidx(iImg),iImg} = ...
        [ '{\bf ' psnrtable{maxidx(iImg),iImg} '}' ];
end

%% Produce LaTeX table
sw = StringWriter();
%
sw.addcr('%#! latex muramatsu')
sw.addcr('%')
sw.addcr('% $Id: main_pkg_psnrs.m 875 2016-01-20 01:51:38Z sho $')
sw.addcr('%')
sw.addcr('\begin{table}[tb]')
sw.addcr('\centering')
sw.add('\caption{PSNRs [dB] of the sparse approximation results for ')
K = params.nCoefsImpl;
strK = [ num2str(floor(K/1000),'%3d') ',' num2str(mod(K,1000),'%3d')];
sw.addcr(sprintf('$%d\\times %d$ pixel images, where $%s$ coefficients remain.',...
    imgSizeApx,imgSizeApx,strK))
sw.addcr(' $\tau$ denotes the number of tree levels.}')
sw.addcr(sprintf('\\label{tab:%02d}',ntab))
sw.addcr('%')
%
sw.addcr('\newcolumntype{Y}{>{\centering\arraybackslash}p{8mm}}') 
sw.add('\begin{tabular}{|c|c||')
for iImg = 1:size(psnrtable,2)
    sw.add('Y|');
end
sw.addcr('} \hline')
%
nCols = length(imgSetLrnNsolt)+2;
%
sw.add('\multicolumn{2}{|c||}{Dictionary}')
for iImg = 1:nImgs
    sw.add([' & {\it ' imgNames{iImg} '}'])
end
sw.addcr(' \\ \hline\hline')
%
sw.add('\multicolumn{2}{|c||}{Sparse K-SVD (')
 sw.add(sprintf('$R=\\frac{%d}{%d}$) }',...
    params.odctsize^2,params.blocksize^2))
for iImg = 1:nImgs
    sw.add([' & ' psnrtable{1,iImg} ' ' ])
end
sw.addcr(' \\ \hline')
%
nChs = sum(params.chs);
for iLv = 1:nLvSet
    if iLv == 1
        sw.add('SGD-NSOLT')
    elseif iLv == 2
        sw.add(sprintf('\\multirow{%d}{*}{',nLvSet-1))
        sw.add(sprintf('($R<\\frac{%d-1}{%d-1}$)',...
            nChs,prod(params.dec)))
        sw.add('}')
    end
    sw.add(sprintf(' & $\\tau=%d$',iLv))
    for iImg = 1:nImgs
        sw.add([' & ' psnrtable{1+iLv,iImg} ' ' ])
    end
    if iLv == nLvSet
        sw.addcr(' \\ \hline')
    else
        sw.addcr(' \\ \cline{2-6}')
    end
end
%
sw.addcr('\end{tabular}')
sw.addcr('\end{table}')
sw.add('\endinput')

disp(sw)
write(sw,sprintf('materials/tab%02d.tex',ntab));

%% Store the source images for approximation and produce LaTeX file
sw = StringWriter();
sw.addcr('%#! latex muramatsu')
sw.addcr('%')
sw.addcr('% $Id: main_pkg_psnrs.m 875 2016-01-20 01:51:38Z sho $')
sw.addcr('%')
sw.addcr('\begin{figure}[tb]')
sw.addcr('\centering')
for iImg = 1:nImgs
    id = ('a'-1)+iImg;
    imgNameApx = imgSetApx{iImg};
    srcImg  = support.fcn_load_testimg2(imgNameApx);    
    imwrite(srcImg,sprintf('tiff/fig%02d%c.tif',nfig,id))
    sw.addcr('\begin{minipage}{.24\linewidth}')    
    sw.add('\centerline{\includegraphics[width=16mm]{')
    sw.add(sprintf('fig%02d%c',nfig,id))
    sw.addcr('}}')
    sw.addcr(sprintf('\\centerline{\\footnotesize (%c)}\\medskip',id))
    sw.addcr('\end{minipage}')    
end
sw.add('\caption{')
sw.add(sprintf('$%d\\times %d$',size(srcImg,1),size(srcImg,2)))
sw.add(' pixel original images in 8-bpp grayscale. ')

for iImg = 1:length(imgNames)
    ids = iImg;
    id = ('a'-1)+ids;
    imgName = imgNames{iImg};    
    sw.add(sprintf('(%c) {\\it %s}. ',id,imgName))
end
sw.addcr('}')

sw.addcr(sprintf('\\label{fig:%02d}',nfig))
sw.addcr('\end{figure}')
sw.add('\endinput')

disp(sw)
write(sw,sprintf('materials/fig%02d.tex',nfig))
