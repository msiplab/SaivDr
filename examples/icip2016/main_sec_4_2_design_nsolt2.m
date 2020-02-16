%MAIN_SEC_4_2_DESIGN_NSOLT2 Dictionary Learning with 2-D NSOLT
%
% This script was used for the design of 2-D NSOLT in Section 4.2.
% The results are stored under sub-folder `results.'
%
% The file name has the following form:
%
%    "nsolt_d..._c..._o..._v..._l..._n..._(imagename)_ufc0.mat"
%
% where the construction parameters are recognized by the rules
%
%  * d(M0)x(M1) : Downsampling factor,
%  * c(ps)+(pa) : # of channels,
%  * o(N0)+(N1) : Polyphase order,
%  * v(nVm)     : # of vanishing moments, no-DC-leakage if nVm = 1,
%  * l(tau)     : # of tree levels,
%  * n(K)       : # of coefficients.
%
% The MAT file containts the following variables.
%
%  * params  : Structure containts design parameters
%  * mses    : Cell array of MSEs observed during iterations
%  * lppufbs : Cell array contains snapshot of NSOLT in every iteration,
%             as an instance object of
%             saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem
%
% The following materials are also reproduced:
%
%  * tiff/fig02x.tif ( x in {a,b,c,d} )
%
% SVN identifier:
% $Id: main_pkg_design_nsolt2.m 875 2016-01-20 01:51:38Z sho $
%
% Requirements: MATLAB R2014a
%
%  * Signal Processing Toolbox
%  * Image Processing Toolbox
%  * Optimization Toolbox
%  * Global Optimization Toolbox
%
% Recommended:
%
%  * MATLAB Coder
%  * Parallel Computing Toolbox
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
% http://msiplab.eng.niigata-u.ac.jp/
%
isEdit = false; % Edit mode

%% Parameter settings
imgSetLrn = { 'goldhill32', 'lena32', 'barbara32', 'baboon32' };
nSubImgs  = 64; % # of patches
picSize   = 32; % Patch size
nTrials   =  1; % # of trials
%
params.nCoefs         = (picSize)^2/8; % # of Coefs.
params.Display        = 'iter';
params.plotFcn        = @optimplotfval;
params.sparseCoding   = 'IterativeHardThresholding';
params.filterDomain   = 'frequency'; % or 'lattice_termination'
%
params.optfcn         = 'fminsgd'; % Stocastic gradient descent
params.useParallel    = 'never';
params.isOptMus       = false;
%params.generationFactorForMus = 2;
params.isFixedCoefs   = true; % FALSE means Simultaneous Optimization
params.nUnfixedInitSteps = 0; % # of unfixed initial steps
params.isMonitoring   = false;
params.isRandomInit   = true;
params.isVisible      = false;
params.isVerbose      = true;
params.stdOfAngRandomInit = 1e-1; 
params.nIters  =  20;
params.maxIter = 128;
params.sgdStep = 'Exponential';
params.sgdStepStart = 16; 
params.sgdStepFinal = 1;
params.sgdGaAngInit = 'off';

%% Condition setting
nDecs = [ 4 4 ];       % Decimation factor [ 4 4 ]
vmSet = { 1 };         % # of Vanishing moments
chSet = { [ 12 12 ] }; % # of channels [ 12 12 ]
odSet = { [ 2 2 ] };   % Polyphase order [ 2 2 ]
lvSet = { 1 };         % Tree levels

%% Pre-build of MEX files
if ~strcmp(params.filterDomain,'frequency')
    for iCh = 1:length(chSet)
        import saivdr.dictionary.nsoltx.mexsrcs.*
        chs = chSet{iCh};
        fcn_autobuild_atomcnc2d([chs(1) chs(2)]);
        fcn_autobuild_atomext2d([chs(1) chs(2)]);
        if chs(1) == chs(2)
            fcn_autobuild_bb_type1(chs(1));
        else
            fcn_autobuild_bb_type2(chs(1),chs(2));
        end
    end
end

%%
nSet = length(imgSetLrn)*length(vmSet)*length(chSet)*length(lvSet);
paramset = cell(nSet,1);
iPar = 1;
%
nImgs = length(imgSetLrn);
orgImgs = cell(nImgs,nSubImgs);
params.dec = nDecs;
for iLv = 1:length(lvSet)
    for iCh = 1:length(chSet)
        for iOrd = 1:length(odSet)
            for iImg = 1:nImgs
                for iVm = 1:length(vmSet)
                    paramset{iPar} = params;
                    paramset{iPar}.nVm = vmSet{iVm};
                    paramset{iPar}.chs = chSet{iCh};
                    paramset{iPar}.nLevels = lvSet{iLv};
                    paramset{iPar}.ord = odSet{iOrd};
                    paramset{iPar}.imgNameLrn = imgSetLrn{iImg};
                    %
                    paramset{iPar}.srcImgs = cell(nSubImgs,1);
                    rng(0,'twister');
                    for iSubImg = 1:nSubImgs
                        subImgName = [imgSetLrn{iImg} 'rnd'];
                        orgPatch = ...
                            im2double(support.fcn_load_testimg2(subImgName));
                        paramset{iPar}.srcImgs{iSubImg} = orgPatch;
                        orgImgs{iImg,iSubImg} = orgPatch;
                    end
                    %
                    iPar = iPar  + 1;
                end
            end
        end
    end
end

%% Dictionary learning
nPars = length(paramset);
for iPar = 1:nPars
    fprintf('Dictionary learning condition: d%dx%d c%d+%d o%d+%d v%d l%d n%d %s...\n',...
        paramset{iPar}.dec(1),paramset{iPar}.dec(2),...
        paramset{iPar}.chs(1),paramset{iPar}.chs(2),...
        paramset{iPar}.ord(1),paramset{iPar}.ord(2),...
        paramset{iPar}.nVm,paramset{iPar}.nLevels,...
        paramset{iPar}.nCoefs,paramset{iPar}.imgNameLrn);
end
%
if ~isEdit
    for iRep = 1:nTrials
        parfor iPar = 1:nPars
            %rng(0,'twister');
            str = support.fcn_updatensolt2(paramset{iPar});
            fprintf('(iRep,iPar) = (%d,%d)\n',iRep,iPar)
        end
    end
end

%% Training images
subOrgImgs = cell(ceil(sqrt(nSubImgs)),ceil(sqrt(nSubImgs)));
for iImg = 1:nImgs
    %
    imgName = imgSetLrn{iImg};
    for iSubImg = 1:nSubImgs
        iRow = floor((iSubImg-1)/(ceil(sqrt(nSubImgs))))+1;
        iCol = mod((iSubImg-1),ceil(sqrt(nSubImgs)))+1;
        subOrgImgs{iRow,iCol} = padarray(orgImgs{iImg,iSubImg},[1 1],1);
    end
    %
    catImg = cell2mat(subOrgImgs);
    subplot(1,nImgs,iImg)
    imshow(catImg)
    ids = iImg;
    id = ('a'-1)+ids;
    fname = sprintf('trnimgs_%c',id);
    imwrite(catImg,sprintf('tiff/%s.tif',fname))
end
