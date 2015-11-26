% MAIN_SWEEPLAMBDAIP Sweep lambda for image inpainting
%
% This script executes exhaustive evaluation of image inpainting
% by sweepig the parameter Lambda in ISTA, which controls the weight
% between the fidelity and sparsity term. Three different dictionaries
% are compared. The results are saved under the folder './results.'
%
% SVN identifier:
% $Id: main_sweeplambdaip.m 683 2015-05-29 08:22:13Z sho $
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

%% Parameters for lambda sweep
nLambdas     = 250;  % # of lambdas
stepLambda   = 5e-5; % Step width of lambda
offsetLambda = 0.0;  % Off set of lambda

%% Load test image
[img, strpartpic] = ...  % Load a test image 
    support.fcn_load_testimg('lena128');
orgImg = im2double(img); % Convert image to double type
nDim = size(orgImg);     % Dimension of original image

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
obsImg = support.fcn_observation(... 
    linproc,orgImg,strpartpic,strlinproc,nsigma);

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
%lppufb = S.lppufb;
lppufb = saivdr.dictionary.utility.fcn_upgrade(S.lppufb);
release(lppufb);
set(lppufb,'OutputMode','ParameterMatrixSet');
synthesizer{iDic} = NsoltFactory.createSynthesis2dSystem(lppufb);
analyzer{iDic}    = NsoltFactory.createAnalysis2dSystem(lppufb);

% Critically-sampled Haar transform
iDic = iDic+1;
strdics{iDic}  = 'CsHaar';
nLevels{iDic} = 6; % # of tree levels
vm = 1;            % # of vanishing moments
sdir = './filters/';
nDec = [ 2 2 ];    % # of decimation factors (My and Mx)
nChs = [ 2 2 ];    % # of channels (ps and pa)
nOrd = [ 0 0 ];    % # of polyphase order (Ny and Nx)
import saivdr.dictionary.nsoltx.*
S = load(sprintf('%s/nsolt_d%dx%d_c%d+%d_o%d+%d_v%d.mat',sdir,...
    nDec(1),nDec(2),nChs(1),nChs(2),nOrd(1),nOrd(2),vm),'lppufb');
%lppufb = S.lppufb;
lppufb = saivdr.dictionary.utility.fcn_upgrade(S.lppufb);
release(lppufb);
set(lppufb,'OutputMode','ParameterMatrixSet');
synthesizer{iDic} = NsoltFactory.createSynthesis2dSystem(lppufb);
analyzer{iDic}    = NsoltFactory.createAnalysis2dSystem(lppufb);

% Undecimated (non-subsampled) Haar transform
iDic = iDic+1;
strdics{iDic}  = 'UdHaar';
nLevels{iDic} = 2;
import saivdr.dictionary.udhaar.*
synthesizer{iDic} = UdHaarSynthesis2dSystem();
analyzer{iDic}    = UdHaarAnalysis2dSystem();

% Union of directional symmetric orthonormal wavelet transforms
iDic = iDic+1;   
strdics{iDic}  = 'UDirSowt';
nLevels{iDic} = 6; % # of tree levels
dec  = [2 2];      % Decimation factor
ord  = [4 4];      % Polyphase order
vm   = 2;          % # of vanishing moments
udsdir = '../dirlot/filters/'; % Folder contains dictionary parameter
import saivdr.dictionary.nsgenlotx.*
import saivdr.dictionary.nsoltx.*
import saivdr.dictionary.mixture.*
isowts = cell(1,5);
fsowts = cell(1,5);
S = load(sprintf('%s/nsgenlot_d%dx%d_o%d+%d_v%d.mat',udsdir,...
    dec(1),dec(2),ord(1),ord(2),vm),'lppufb');
%lppufb = S.lppufb;
lppufb = saivdr.dictionary.utility.fcn_upgrade(S.lppufb);
release(lppufb);
set(lppufb,'OutputMode','ParameterMatrixSet');
isowts{1} = NsoltFactory.createSynthesis2dSystem(lppufb);
fsowts{1} = NsoltFactory.createAnalysis2dSystem(lppufb);
phiset = [ -30 30 60 120 ];
idx = 2;
for phi = phiset
    S = load(sprintf('%s/dirlot_d%dx%d_o%d+%d_tvm%06.2f.mat',udsdir,...
        dec(1),dec(2),ord(1),ord(2),phi),'lppufb');
    %lppufb = S.lppufb;
    lppufb = saivdr.dictionary.utility.fcn_upgrade(S.lppufb);
    release(lppufb);
    set(lppufb,'OutputMode','ParameterMatrixSet');
    isowts{idx} = NsoltFactory.createSynthesis2dSystem(lppufb);
    fsowts{idx} = NsoltFactory.createAnalysis2dSystem(lppufb);
    idx = idx + 1;
end
synthesizer{iDic} = MixtureOfUnitarySynthesisSystem(...
    'UnitarySynthesizerSet',isowts);
analyzer{iDic} = MixtureOfUnitaryAnalysisSystem(...
    'UnitaryAnalyzerSet',fsowts);

%
nDics = iDic;

%% Create step monitor
isVerbose = false; 
isVisible = false; 
import saivdr.utility.StepMonitoringSystem
stepmonitor = StepMonitoringSystem(...
    'SourceImage',orgImg,...
    'MaxIter', maxIter,...
    'IsMSE',  true,...
    'IsPSNR', true,...
    'IsSSIM', true,...
    'IsVisible', isVisible,...
    'IsVerbose', isVerbose);

%% ISTA
stralg = 'ISTA';
lambda_ = 0;

import saivdr.restoration.ista.IstaImRestoration
rstrset = cell(nDics,1);
for iDic = 1:nDics
    rstrset{iDic} = IstaImRestoration(...
        'Synthesizer',synthesizer{iDic},...
        'AdjOfSynthesizer',analyzer{iDic},...
        'LinearProcess',linproc,...
        'NumberOfTreeLevels',nLevels{iDic},...
        'Lambda',lambda_);
    set(rstrset{iDic},'MaxIter',maxIter);
    set(rstrset{iDic},'Eps0',eps0);
end

%% Sweep lambda for evaluating
for iDic = 1:nDics
    psnrswp = zeros(1,nLambdas);
    ssimswp = zeros(1,nLambdas);
    rstr   = rstrset{iDic};
    strdic = strdics{iDic}; 
    parfor iLambda = 1:nLambdas
        lambda_ = (iLambda-1)*stepLambda + offsetLambda;
        rstr_ = clone(rstr);
        stepmonitor_ = clone(stepmonitor);
        set(rstr_,'StepMonitor',stepmonitor_);
        set(rstr_,'Lambda',lambda_);
        % Restoration
        step(rstr_,obsImg);
        % Extract the final result
        nItr   = get(stepmonitor_,'nItr');
        mses_  = get(stepmonitor_,'MSEs');
        psnrs_ = get(stepmonitor_,'PSNRs');
        ssims_ = get(stepmonitor_,'SSIMs');
        % Extract the final values
        psnrswp(iLambda) = psnrs_(nItr);
        mseswp(iLambda) = mses_(nItr);
        ssimswp(iLambda) = ssims_(nItr);
        fprintf('(%s) lambda = %7.4f : nItr = %4d, psnr = %6.2f [dB], ssim = %6.3f\n',...
            strdic, lambda_, nItr, psnrs_(nItr), ssims_(nItr));
    end
    
    %% Extract the best result
    lambda = (0:nLambdas-1)*stepLambda + offsetLambda;
    % Search Max. values
    [maxpsnr,idxpsnrmax] = max(psnrswp(:));
    [maxssim,idxssimmax] = max(ssimswp(:));
    lambdamaxpsnr = lambda(idxpsnrmax);
    lambdamaxssim = lambda(idxssimmax);
    % Update the restoration system by the best value of lambda 
    set(rstr,'Lambda',lambdamaxpsnr);
    % String for identifing the condtion 
    scnd = sprintf('%s_%s_%s_%s_ns%06.2f',...
        strpartpic,lower(stralg),lower(strdic),strlinproc,nsigma);
    % File name
    psnrsname = sprintf('./results/psnr_ssim_%s.mat',scnd);
    disp(psnrsname)
    % Save data
    save(psnrsname,'lambda','psnrswp','mseswp','ssimswp','rstr',...
        'maxpsnr','lambdamaxpsnr','maxssim','lambdamaxssim')
    fprintf('*(%s) lambda = %7.5f : max psnr = %6.2f [dB]\n',...
        strdic, lambdamaxpsnr, maxpsnr);
end
