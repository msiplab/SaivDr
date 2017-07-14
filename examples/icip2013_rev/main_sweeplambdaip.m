% MAIN_SWEEPLAMBDAIP Sweep lambda for image inpainting
%
% This script executes exhaustive evaluation of image inpainting
% by sweepig the parameter Lambda in ISTA, which controls the weight
% between the fidelity and sparsity term. Two different dictionaries
% are compared. The results are saved under the folder './results.'
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
clc

%% Parameters for lambda sweep
nLambdas     = 200;  % # of lambdas
stepLambda   = 5e-5; % Step width of lambda
offsetLambda = 0.0;  % Off set of lambda

%% Load test images
imgset = { 'goldhill128', 'lena128', 'barbara128', 'baboon128' };
nImgs  = length(imgset);
strpartpic = cell(1,nImgs);
orgImg     = cell(1,nImgs);
nDim       = [ 128 128 ]; % Dimension of original image
for iImg = 1:nImgs
    [img, strpartpic{iImg}] = ...    % Load a test image
        support.fcn_load_testimg(imgset{iImg});
    orgImg{iImg} = im2double(img);   % Convert image to double type
    if size(img,1) ~= nDim(1) && size(img,2) ~= nDim(2)
        error('Original images must be of size %d x %d\n',nDim(1),nDim(2));
    end
end

%% Parameters for ISTA
maxIter = 1000; % Max. # of iterations
eps0 = 1e-7; % Tolerance value for convergence

%% Parameters for degradation
% Inpainting
losstype = 'Random'; % Type of pixel loss
density  = 0.2;      % Pixel loss density
seed     = 0;        % Seed for RNG function
nsigma   = 0;        % Std. deviation of noise

%% Create degradation linear process
import saivdr.degradation.linearprocess.*
linproc = PixelLossSystem(... % Object for pixel-loss process
    'LossType',losstype,...
    'Density',density,...
    'Seed',seed);

% String for identificatin of linear process
strlinproc = support.fcn_make_strlinproc(linproc);

% Use file for lambda_max
fname_lmax = sprintf('./lmax/%s_%dx%d.mat',strlinproc,nDim(1),nDim(2));
set(linproc,...              % The way of managing max. eigen value
    'UseFileForLambdaMax',true,...
    'FileNameForLambdaMax',fname_lmax);

%% Load or create an observed image
obsImg = cell(1,nImgs);
for iImg = 1:nImgs
    obsImg{iImg} = support.fcn_observation(...
        linproc,orgImg{iImg},strpartpic{iImg},strlinproc,nsigma);
end

%% Create dictionaries
iDic = 0;

% NSOLT
iDic = iDic+1;
strdics{iDic}  = 'Nsolt52';
nLevels{iDic} = 6; % # of tree levels
vm = 1;            % # of vanishing moments
sdir = './filters/';
nDec = [ 2 2 ];    % # of decimation factors (My and Mx)
nChs = [ 5 2 ];    % # of channels (ps and pa)
nOrd = [ 4 4 ];    % # of polyphase order (Ny and Nx)
import saivdr.dictionary.nsoltx.*
S = load(sprintf('%s/nsolt_d%dx%d_c%d+%d_o%d+%d_v%d.mat',sdir,...    
    nDec(1),nDec(2),nChs(1),nChs(2),nOrd(1),nOrd(2),vm),'lppufb');
lppufb = saivdr.dictionary.utility.fcn_upgrade(S.lppufb);
release(lppufb);
set(lppufb,'OutputMode','ParameterMatrixSet');
synthesizers{iDic} = NsoltFactory.createSynthesis2dSystem(lppufb);
analyzers{iDic}    = NsoltFactory.createAnalysis2dSystem(lppufb);

% Undecimated (non-subsampled) Haar transform
iDic = iDic+1;
strdics{iDic}  = 'UdHaar';
nLevels{iDic} = 2;
import saivdr.dictionary.udhaar.*
synthesizers{iDic} = UdHaarSynthesis2dSystem();
analyzers{iDic}    = UdHaarAnalysis2dSystem();

%
nDics = iDic;

%% Create step monitor
isVerbose = false;
isVisible = false;
import saivdr.utility.StepMonitoringSystem
stepmonitor = StepMonitoringSystem(...
    'DataType','Image',...
    'MaxIter', maxIter,...
    'IsMSE',  true,...
    'IsPSNR', true,...
    'IsSSIM', true,...
    'IsVisible', isVisible,...
    'IsVerbose', isVerbose);

%% ISTA
stralg = 'ISTA';
lambda_ = 0;

import saivdr.restoration.ista.IstaImRestoration2d
rstrset = cell(nDics,1);
for iDic = 1:nDics
    rstrset{iDic} = IstaImRestoration2d(...
        'Synthesizer',synthesizers{iDic},...
        'AdjOfSynthesizer',analyzers{iDic},...
        'LinearProcess',linproc,...
        'NumberOfTreeLevels',nLevels{iDic},...
        'Lambda',lambda_);
    set(rstrset{iDic},'MaxIter',maxIter);
    set(rstrset{iDic},'Eps0',eps0);
end

%% Sweep lambda 
params.orgImgSet    = orgImg;
params.obsimgSet    = obsImg;
params.strImgSet    = strpartpic;
params.imrstrSet    = rstrset;
params.stralg       = stralg;
params.strDicSet    = strdics;
params.strlinproc   = strlinproc;
params.nImgs        = nImgs;
params.nDics        = nDics;
params.nLambdas     = nLambdas;
params.stepLambda   = stepLambda;
params.offsetLambda = offsetLambda;
params.stepmonitor  = stepmonitor;
params.nsigma       = nsigma;
%
support.fcn_sweeplambda(params);
