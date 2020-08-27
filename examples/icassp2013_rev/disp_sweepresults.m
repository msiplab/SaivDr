% DISP_SWEEPRESULTS Show sweep results of Lambda vs PSNR/SSIM 
%
% This script shows the results produced by MAIN_SWEEPLAMBDAIP
% in plot graphs. Thus, it is required to execute 
% MAIN_SWEEPLAMBDAIP previously.
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014-2017, Shogo MURAMATSU
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

%% Load test image
[~, strpartpic] = support.fcn_load_testimg('lena128');

%% Parameters for degradation
% Inpainting
losstype = 'Random';
density  = 0.2;
seed     = 0;
nsigma   = 0;

import saivdr.degradation.linearprocess.*
linproc = PixelLossSystem(...
    'LossType',losstype,...
    'Density',density,...
    'Seed',seed);

% String for identificatin of linear process
strlinproc = support.fcn_make_strlinproc(linproc); 

%% Dictionary
iDic = 0;

% Type-II NSOLT 
iDic = iDic+1;
strdics{iDic} = 'Nsolt52';

% Critically-sampled Haar transform
iDic = iDic+1;
strdics{iDic} = 'CsHaar';

% Undecimated Haar transform
iDic = iDic+1;
strdics{iDic} = 'UdHaar';

% Union of directional symmetric orthonormal wavelet transforms
iDic = iDic+1;
strdics{iDic} = 'UDirSowt';

%% Restoration algorithm
stralg = 'ISTA';

%% Show graphs
for iDic = 1:length(strdics)
    scnd = sprintf('%s_%s_%s_%s_ns%06.2f',...
        strpartpic,lower(stralg),lower(strdics{iDic}),strlinproc,nsigma);
    fname = sprintf('./results/psnr_ssim_%s.mat',scnd);
    h = figure;
    figname = sprintf('%s_%s_%s',...
        strpartpic,lower(strdics{iDic}),strlinproc);
    set(h,'Name',figname)
    if exist(fname,'file') == 2
        % Load data
        S = load(fname,'-mat','lambda','psnrswp','maxpsnr',...
            'lambdamaxpsnr','ssimswp','maxssim','lambdamaxssim');
        lambda = S.lambda;
        psnrswp = S.psnrswp;
        maxpsnr = S.maxpsnr;
        lambdamaxpsnr = S.lambdamaxpsnr;
        ssimswp = S.ssimswp;
        maxssim = S.maxssim;
        lambdamaxssim = S.lambdamaxssim;
        % Plot PSNRs
        subplot(1,2,1)
        plot(lambda,psnrswp)
        grid on
        axis([min(lambda(:)) max(lambda(:)) 0 40 ])
        xlabel('\lambda')                
        ylabel('PSNR [dB]')
        line([lambdamaxpsnr lambdamaxpsnr],[0 maxpsnr],'Color','red')
        text(lambdamaxpsnr+0.001,1,num2str(lambdamaxpsnr))
        text(lambdamaxpsnr,maxpsnr+1,num2str(maxpsnr))
        % Plot SSIMs
        subplot(1,2,2)
        plot(lambda,ssimswp)
        axis([min(lambda(:)) max(lambda(:)) 0 1])
        grid on
        xlabel('\lambda')
        ylabel('SSIM [-]')
        line([lambdamaxssim lambdamaxssim],[0 maxssim],'Color','red')
        text(lambdamaxssim+0.001,0.025,num2str(lambdamaxssim))
        text(lambdamaxssim,maxssim+0.025,num2str(maxssim))        
        % Show maximun psnr and ssim
        fprintf('%s\n',fname);
        fprintf('MAX PSNR = %6.3f ( %7.5f )\n', maxpsnr, lambdamaxpsnr)
        fprintf('MAX SSIM = %6.4f ( %7.5f )\n', maxssim, lambdamaxssim)
    else
        fprintf('%s does not exist...\n',fname);
    end
end
