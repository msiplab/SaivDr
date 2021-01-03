% MAIN_NSOLTIMDB Image Deblurring with Type-II NSOLT
%
% This script executes image debluring with ISTA and Type-II NSOLT
% designed by using MAIN_PARNSOLTDSGN. The design data placed under
% the folder './filters' is loaded.
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014-2020, Shogo MURAMATSU
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
clc

%% Parameter setting for image restoration

% Parameters for degradation
blurtype = 'Gaussian'; % Blur type
hsigma   = 2;          % Sigma for Gausian kernel
nsigma   = 5;          % Sigma for AWGN

% Parameters for dictionary
strdic  = 'Nsolt52';
nlevels = 6;         % # of wavelet tree levels
dec  = [2 2];        % Decimation factor
nChs = [5 2];        % # of channels
ord  = [4 4];        % Polyphase order
vm   = 1;            % # of vanishing moments
sdir = './filters/'; % Folder contains dictionary parameter

% Parameter for ISTA
lambda =  0.00320;   % Lambda
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
linproc = BlurSystem(...
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
import saivdr.dictionary.nsoltx.*
s = load(sprintf('%s/nsolt_d%dx%d_c%d+%d_o%d+%d_v%d.mat',sdir,...
    dec(1),dec(2),nChs(1),nChs(2),ord(1),ord(2),vm),'lppufb');
lppufb = saivdr.dictionary.utility.fcn_upgrade(s.lppufb);
release(lppufb);
set(lppufb,'OutputMode','ParameterMatrixSet');
synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb);
analyzer = NsoltFactory.createAnalysis2dSystem(lppufb,...
    'NumberOfLevels',nlevels);

%% Create a step monitor
import saivdr.utility.StepMonitoringSystem
hfig1 = figure(1);
stepmonitor = StepMonitoringSystem(...
    'DataType','Image',...
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
import saivdr.restoration.ista.IstaImRestoration2d
rstr = IstaImRestoration2d(...
    'Synthesizer',synthesizer,...
    'AdjOfSynthesizer',analyzer,...
    'LinearProcess',linproc,...
    ...'NumberOfTreeLevels',nlevels,...
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
mse = mses_(1:nItr);   
psnr = psnrs_(1:nItr); 
ssim = ssims_(1:nItr); 
s = sprintf('%s_%s_%s_%s_ns%06.2f',...
    strpartpic,lower(stralg),lower(strdic),strlinproc,nsigma);
imwrite(resImg,sprintf('./results/res_%s.tif',s));
save(sprintf('./results/eval_%s.mat',s),'nItr','psnr','mse','ssim')

%% Wiener
stralg = 'Wiener';
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
blurKernel = get(linproc,'BlurKernel');
nsr = (nsigma/255)^2 / var(orgImg(:));

tic
wnfImg = deconvwnr(obsImg, blurKernel, nsr);
[mse, psnr,ssim] = step(stepmonitor,wnfImg);
toc
%
s = sprintf('%s_%s_%s_ns%06.2f',strpartpic,lower(stralg),strlinproc,nsigma);
imwrite(wnfImg,sprintf('./results/res_%s.tif',s));
save(sprintf('./results/eval_%s.mat',s),'psnr','mse','ssim')
