%MAIN_UDHAARIMSR Image Super-resolution with undecimated Haar transform
%
% This script executes image debluring with ISTA and undecimated Haar
% transform.
%
% Requirements: MATLAB R2013b
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
% LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
%
clc

%% Parameter setting for image restoration

% Parameters for degradation
dFactor  = [2 2];      % Decimation factor 
blurtype = 'Gaussian'; % Blur type
hsigma   = 2;          % Sigma for Gausian kernel
nsigma   = 0;          % Sigma for AWGN

% Parameters for dictionary
strdic  = 'UdHaar'; 
nlevels = 2;         % # of wavelet tree levels

% Parameter for ISTA
lambda =  0.00016;   % Lambda 
maxIter = 1000;         % Maximum number of iterations
eps0 = 1e-7;         % Criteria for convergence
isverbose = true;    % Verbose mode
isvisible = true;    % Monitor intermediate results 

%% Load test image
[img,strpartpic] = support.fcn_load_testimg('barbara128');
orgImg = im2double(img);
nDim = size(orgImg);

%% Create degradation linear process
import saivdr.degradation.linearprocess.*
linproc = DecimationSystem(...
    'VerticalDecimationFactor',dFactor(1),...
    'HorizontalDecimationFactor',dFactor(2),...
    'BlurType',blurtype,...
    'SigmaOfGaussianKernel',hsigma);

% String for identificatin of linear process
strlinproc = support.fcn_make_strlinproc(linproc); 

% Use file for lambda_max
fname_lmax = sprintf('./lmax/%s_%dx%d.mat',strlinproc,nDim(1),nDim(2));
set(linproc,...
    'UseFileForLambdaMax',true,...
    'FileNameForLambdaMax',fname_lmax);

%% Load or create an observed image
obsImg = support.fcn_observation(...
    linproc,orgImg,strpartpic,strlinproc,nsigma);

%% Create a dictionary
import saivdr.dictionary.udhaar.*
synthesizer = UdHaarSynthesis2dSystem();
analyzer    = UdHaarAnalysis2dSystem();

%% Create a step monitor
import saivdr.utility.StepMonitoringSystem
hfig1 = figure(1);
stepmonitor = StepMonitoringSystem(...
    'SourceImage',orgImg,...
    'ObservedImage',obsImg,...
    'MaxIter', maxIter,...
    'IsMSE', true,...
    'IsPSNR', true,...
    'IsSSIM', true,...
    'IsVisible', isvisible,...
    'ImageFigureHandle',hfig1,...
    'IsVerbose', isverbose);

%% ISTA
stralg = 'ISTA';
fprintf('\n%s',stralg)
import saivdr.restoration.ista.IstaImRestoration
rstr = IstaImRestoration(...
    'Synthesizer',synthesizer,...
    'AdjOfSynthesizer',analyzer,...
    'LinearProcess',linproc,...
    'NumberOfTreeLevels',nlevels,...
    'Lambda',lambda);
set(rstr,'MaxIter',maxIter);
set(rstr,'Eps0',eps0);  
set(rstr,'StepMonitor',stepmonitor);
set(hfig1,'Name',[stralg ' ' strdic])

tic
resImg = step(rstr,obsImg);
toc

%% Save results
nItr   = get(stepmonitor,'nItr');
mses_  = get(stepmonitor,'MSEs');
psnrs_ = get(stepmonitor,'PSNRs');
ssims_ = get(stepmonitor,'SSIMs');
mse = mses_(1:nItr);   %#ok
psnr = psnrs_(1:nItr); %#ok
ssim = ssims_(1:nItr); %#ok
s = sprintf('%s_%s_%s_%s_ns%06.2f',...
    strpartpic,lower(stralg),lower(strdic),strlinproc,nsigma);
imwrite(resImg,sprintf('./results/res_%s.tif',s));
save(sprintf('./results/eval_%s.mat',s),'nItr','psnr','mse','ssim')

%% Bicubic
stralg = 'Bicubic';
fprintf('\n%s',stralg)
%
hfig2 = figure(2);
stepmonitor = StepMonitoringSystem(...
    'SourceImage',orgImg,...
    'ObservedImage',obsImg,...
    'MaxIter', 1,...
    'IsMSE', true,...
    'IsPSNR', true,...
    'IsSSIM', true,...
    'IsVisible', true,...
    'ImageFigureHandle',hfig2,...
    'IsVerbose', isverbose);
set(hfig2,'Name',stralg)
%
tic
bicImg = imresize(obsImg, dFactor.*size(obsImg), 'bicubic');
[mse, psnr,ssim] = step(stepmonitor,bicImg);
toc
%
s = sprintf('%s_%s_%s_ns%06.2f',strpartpic,lower(stralg),strlinproc,nsigma);
imwrite(bicImg,sprintf('./results/res_%s.tif',s));
save(sprintf('./results/eval_%s.mat',s),'psnr','mse','ssim')
