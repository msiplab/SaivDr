% DISP_SWEEPRESULTS Show sweep results of Lambda vs PSNR/SSIM
%
% This script shows the results produced by MAIN_SWEEPLAMBDADB,
% MAIN_SWEEPLMBDASR and MAIN_SWEEPLAMBDAIP. Thus, it is required to
% excecute these scripts previously.
%
% SVN identifier:
% $Id: disp_sweepresults.m 683 2015-05-29 08:22:13Z sho $
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014-2015, Shogo MURAMATSU
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
clear all; close all; clc

%% Prepare string set for test images
imgset = { 'goldhill128', 'lena128', 'barbara128', 'baboon128' };
nImgs  = length(imgset);
strpartpic = cell(1,nImgs);
nDim       = [ 128 128 ]; % Dimension of original image
for iImg = 1:nImgs
    [~, strpartpic{iImg}] = ...
        support.fcn_load_testimg(imgset{iImg});
end

%% Parameters for degradation
import saivdr.degradation.linearprocess.*
iLp = 0;

% Deblur
iLp = iLp + 1;
blurtype = 'Gaussian'; % Blur type
hsigma   = 2;          % Sigma for Gausian kernel
linproc =  BlurSystem(...
    'BlurType',blurtype,...
    'SigmaOfGaussianKernel',hsigma);
strlinproc{iLp} = support.fcn_make_strlinproc(linproc);
nsigma{iLp}   = 5;

% Super-resolution
iLp = iLp + 1;
dFactor  = [2 2];      % Decimation factor
blurtype = 'Gaussian'; % Blur type
hsigma   = 2;          % Sigma for Gausian kernel
linproc = DecimationSystem(...
    'VerticalDecimationFactor',dFactor(1),...
    'HorizontalDecimationFactor',dFactor(2),...
    'BlurType',blurtype,...
    'SigmaOfGaussianKernel',hsigma);
strlinproc{iLp} = support.fcn_make_strlinproc(linproc);
nsigma{iLp}   = 0;

% Inpainting
iLp = iLp + 1;
losstype = 'Random';
density  = 0.2;
seed     = 0;
linproc = PixelLossSystem(...
    'LossType',losstype,...
    'Density',density,...
    'Seed',seed);
strlinproc{iLp} = support.fcn_make_strlinproc(linproc);
nsigma{iLp}   = 0;

%
nLps = iLp;

%% Dictionary
iDic = 0;

% Undecimated Haar transform
iDic = iDic+1;
nLevelsNsHaarWt = 2;
strdics{iDic} = 'UdHaar';

% Type-II NSOLT
iDic = iDic+1;
strdics{iDic} = 'Nsolt52';

%
nDics = iDic;

%% Restoration algorithm
stralg = 'ISTA';

%% Show graphs and generate tables
psnrtcell = cell(nLps,nImgs,nDics);
plmdtcell = cell(nLps,nImgs,nDics);
ssimtcell = cell(nLps,nImgs,nDics);
slmdtcell = cell(nLps,nImgs,nDics);
for iImg = 1:nImgs
    for iLp = 1:nLps
        for iDic = 1:length(strdics)
            scnd = sprintf('%s_%s_%s_%s_ns%06.2f',...
                strpartpic{iImg},lower(stralg),lower(strdics{iDic}),...
                strlinproc{iLp},nsigma{iLp});
            fname = sprintf('./results/psnr_ssim_%s.mat',scnd);
            h = figure;
            figname = sprintf('%s_%s_%s',...
                strpartpic{iImg},lower(strdics{iDic}),strlinproc{iLp});            
            set(h,'Name',figname)
            if exist(fname,'file') == 2
                % Load data
                S =  load(fname,'-mat');
                lambda = S.lambda;
                psnrswp = S.psnrswp;
                maxpsnr = S.maxpsnr;
                lambdamaxpsnr = S.lambdamaxpsnr;
                ssimswp = S.ssimswp;
                maxssim = S.maxssim;
                lambdamaxssim = S.lambdamaxssim;
                % Plot PSNRs
                figure(h)
                subplot(1,2,1)
                plot(lambda,psnrswp)
                grid on
                axis([min(lambda(:)) max(lambda(:)) 0 40 ])
                xlabel('\lambda')
                ylabel('PSNR [dB]')
                line([lambdamaxpsnr lambdamaxpsnr],[0 maxpsnr],'Color','red')
                text(lambdamaxpsnr+lambda(4),1,num2str(lambdamaxpsnr))
                text(lambdamaxpsnr,maxpsnr+1,num2str(maxpsnr))
                % Plot SSIMs
                subplot(1,2,2)
                plot(lambda,ssimswp)
                axis([min(lambda(:)) max(lambda(:)) 0 1])
                grid on
                xlabel('\lambda')
                ylabel('SSIM [-]')
                line([lambdamaxssim lambdamaxssim],[0 maxssim],'Color','red')
                text(lambdamaxssim+lambda(4),0.025,num2str(lambdamaxssim))
                text(lambdamaxssim,maxssim+0.025,num2str(maxssim))
                % Show maximun psnr and ssim
                fprintf('%s\n',fname);
                fprintf('MAX PSNR = %6.3f ( %7.5f )\n', maxpsnr, lambdamaxpsnr)
                fprintf('MAX SSIM = %6.4f ( %7.5f )\n', maxssim, lambdamaxssim)
                % Generate tables
                psnrtcell{iLp,iImg,iDic} = maxpsnr;
                plmdtcell{iLp,iImg,iDic} = lambdamaxpsnr;
                ssimtcell{iLp,iImg,iDic} = maxssim;
                slmdtcell{iLp,iImg,iDic} = lambdamaxssim;
            else
                fprintf('%s does not exist...\n',fname);
            end
        end
    end
end

%% PSNR
disp('PSNR')
for iLp = 1:nLps
    fprintf('%s\n',strlinproc{iLp})
    fprintf('             ')    
    for iDic = 1:nDics
        fprintf('%s          ',strdics{iDic})
    end
    fprintf('\n')
    for iImg = 1:nImgs
        fprintf('%12s ',imgset{iImg})
        for iDic = 1:nDics
            fprintf('%5.2f (%7.5f) ',psnrtcell{iLp,iImg,iDic},...
                plmdtcell{iLp,iImg,iDic});
        end
        fprintf('\n')
    end
    fprintf('\n')
end

%% SSIM
disp('SSIM')
for iLp = 1:nLps
    fprintf('%s\n',strlinproc{iLp})
    fprintf('             ')
    for iDic = 1:nDics
        fprintf('%s          ',strdics{iDic})
    end
    fprintf('\n')
    for iImg = 1:nImgs
        fprintf('%12s ',imgset{iImg})
        for iDic = 1:nDics
            fprintf('%5.3f (%7.5f) ',ssimtcell{iLp,iImg,iDic},...
                slmdtcell{iLp,iImg,iDic});
        end
        fprintf('\n')
    end
    fprintf('\n')
end
