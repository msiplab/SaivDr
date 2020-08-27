%% Read a source image

download_lena
srcImg = imread('lena.png');
width  = 256; % Width
height = 256; % Height
px     = 128;  % Horizontal position of cropping
py     = 128;  % Vertical position of cropping
orgImg = im2double(srcImg(py:py+height-1,px:px+width-1,:));

%% Create a degradation system object
import saivdr.degradation.linearprocess.PixelLossSystem
losstype = 'Random'; % Pixel loss type
density  = 0.4;      % Pixel loss density 
seed     = 0;        % Random seed for pixel loss
linproc = PixelLossSystem(...
    'LossType',losstype,...
    'Density',density,...
    'Seed',seed);

import saivdr.degradation.noiseprocess.AdditiveWhiteGaussianNoiseSystem
nsigma    = 0;              % Sigma for AWGN for scale [0..255]
noise_var = (nsigma/255)^2; % Normalize sigma to scale [0..1]
awgn = AdditiveWhiteGaussianNoiseSystem(... % Instantiation of AWGN
    'Mean',     0,...
    'Variance', noise_var);

import saivdr.degradation.DegradationSystem
dgrd = DegradationSystem(... % Integration of blur and AWGN
    'LinearProcess', linproc,...
    'NoiseProcess',  awgn);

%% Generate an observed image

obsImg = dgrd.step(orgImg);

%% Create an NSOLT system object

% Parameters for NSOLT
nLevels = 4;     % # of wavelet tree levels
nDec    = [2 2]; % Decimation factor
nChs    = [4 4]; % # of channels
nOrd    = [4 4]; % Polyphase order
nVm     = 1;     % # of vanishing moments

% Location which containts a pre-designed NSOLT
sdir = '../icassp2014/results';

% Load a pre-designed dictionary from a MAT-file
s = load(sprintf('%s/nsolt_d%dx%d_c%d+%d_o%d+%d_v%d_l%d_n%d_%s.mat',...
    sdir,nDec(1),nDec(2),nChs(1),nChs(2),nOrd(1),nOrd(2),nVm,nLevels,...
    2048,'lena128'));
nsolt = s.lppufbs{end}; % saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIVm1System
nsolt = saivdr.dictionary.utility.fcn_upgrade(nsolt); 
    
% Show the atomic images by using a method atmimshow()
hfig1 = figure(1);
nsolt.atmimshow()
hfig1.Name = 'Atomic images of NSOLT';

% In order to set the object to synthesis and analysis system object,
% change the output mode to 'ParameterMatrixSet.'
nsolt.release()
nsolt.OutputMode = 'ParameterMatrixSet';

%% Create an analysis and synthesis system object
% Since the object of OvsdLpPuFb2dTypeIVm1System, _nsolt_ , is not able 
% to process images by itself, we have to construct an analysis and 
% synthesis system for analyzing and synthesizing an image, respectively.
% The following two systems can do these tasks:
%
% * saivdr.dictionary.generalfb.Synthesis2dSystem
% * saivdr.dictionary.generalfb.Analysis2dSystem

import saivdr.dictionary.generalfb.Analysis2dSystem
import saivdr.dictionary.generalfb.Synthesis2dSystem

% Change the output mode of NSOLT to 'AnalysisFilters' and
% draw inpulse responses of the analysis filters.
nsolt.release()
nsolt.OutputMode = 'AnalysisFilters';
analysisFilters = nsolt.step([],[]);

% Change the output mode of NSOLT to 'SynthesisFilters' and
% draw inpulse responses of the synthesis filters.
nsolt.release()
nsolt.OutputMode = 'SynthesisFilters';
synthesisFilters = nsolt.step([],[]);

% Create analysis ans synthesis system objects with
% frequency domain filtering mode.
analyzer    = Analysis2dSystem(...
    'DecimationFactor',nDec,...
    'AnalysisFilters',analysisFilters,...
    'NumberOfLevels', nLevels,... 
    'FilterDomain','Frequency');
analyzer.UseGpu = false;
synthesizer = Synthesis2dSystem(...
    'DecimationFactor',nDec,...
    'SynthesisFilters',synthesisFilters,...
    'FilterDomain','Frequency');
setFrameBound(synthesizer,1);
synthesizer.UseGpu = false;

%% Create an ISTA-based image restoration system object

% Instantiation of ISTA system object
import saivdr.restoration.ista.IstaImRestoration2d
lambda    = 0.01;                  % lambda
ista = IstaImRestoration2d(...
    'Synthesizer',        synthesizer,... % Synthesizer (Dictionary)
    'AdjOfSynthesizer',   analyzer,...    % Analyzer (Adj. of dictionary)
    'LinearProcess',      linproc,...        % Blur process
    'Lambda',             lambda);        % Parameter lambda

%% Create a step monitor system object

% Parameters for StepMonitoringSystem
isverbose = true;  % Verbose mode
isvisible = true;  % Monitor intermediate results
hfig2 = figure(2); % Figure to show the source, observed and result image 
hfig2.Name = 'ISTA-based Image Restoration';

% Instantiation of StepMonitoringSystem
import saivdr.utility.StepMonitoringSystem
stepmonitor = StepMonitoringSystem(...
    'SourceImage',   orgImg,...    % Original image
    'ObservedImage', obsImg,...    % Observed image
    'IsMSE',         false,...     % Switch for MSE  evaluation
    'IsPSNR',        true,...      % Switch for PSNR evaluation
    'IsSSIM',        false,...     % Switch for SSIM evaluation
    'IsVerbose',     isverbose,... % Switch for verbose mode
    'IsVisible',     isvisible,... % Switch for display intermediate result
    'ImageFigureHandle',hfig2);    % Figure handle
    
% Set the object to the ISTA system object
ista.StepMonitor = stepmonitor;

%% Perform ISTA-based image restoration

fprintf('\n ISTA')
resImg = ista.step(obsImg); % STEP method of IstaImRestoration

%% Extract the final evaluation  

nItr  = stepmonitor.nItr;
psnrs = stepmonitor.PSNRs;
psnr_ista = psnrs(nItr);

%%
clear psnr
%imwrite(orgImg,'org.tif');
imwrite(obsImg,[strrep(sprintf('obs_ip_%05.2f',psnr(orgImg,obsImg)),'.','_'),'.png']);
imwrite(resImg,[strrep(sprintf('res_ip_%05.2f',psnr(orgImg,resImg)),'.','_'),'.png']);
