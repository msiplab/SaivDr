setpath

srcImg = imread('peppers.png');
width  = 256; % Width
height = 256; % Height
px     = 64;  % Horizontal position of cropping
py     = 64;  % Vertical position of cropping
orgImg = im2double(srcImg(py:py+height-1,px:px+width-1,:));

obsImg = orgImg;

% Parameters for NSOLT
%nLevels = 4;     % # of wavelet tree levels
nLevels = 1;
nDec    = [2 2]; % Decimation factor
nChs    = [4 4]; % # of channels
nOrd    = [2 2]; % Polyphase order
%nOrd = [0 0];
nVm     = 0;     % # of vanishing moments

% Location which containts a pre-designed NSOLT
sdir = './examples/quickdesign/results';

% % Load a pre-designed dictionary from a MAT-file
% s = load(sprintf('%s/nsolt_d%dx%d_c%d+%d_o%d+%d_v%d_l%d_n%d_%s.mat',...
%     sdir,nDec(1),nDec(2),nChs(1),nChs(2),nOrd(1),nOrd(2),nVm,nLevels,...
%     2048,'peppers128x128'),'nsolt');
% nsolt = s.nsolt; % saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIVm1System

% angsV0 = 2*pi*rand(28,1);
% angsWx1 = 2*pi*rand(6,1);
% angsUx1 = 2*pi*rand(6,1);
% angsBx1 = pi/4*ones(floor(sum(nChs)/4),1);
% angsWx2 = 2*pi*rand(6,1);
% angsUx2 = 2*pi*rand(6,1);
% angsBx2 = pi/4*ones(floor(sum(nChs)/4),1);
% angsWy1 = 2*pi*rand(6,1);
% angsUy1 = 2*pi*rand(6,1);
% angsBy1 = pi/4*ones(floor(sum(nChs)/4),1);
% angsWy2 = 2*pi*rand(6,1);
% angsUy2 = 2*pi*rand(6,1);
% angsBy2 = pi/4*ones(floor(sum(nChs)/4),1);
% 
% 
% angles = [angsV0;angsWx1;angsUx1;angsBx1;angsWx2;angsUx2;angsBx2;angsWy1;angsUy1;angsBy1;angsWy2;angsUy2;angsBy2];
% mus = ones(8,5);
% angles = angsV0;
% mus = ones(8,1);

nsolt = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
    'DecimationFactor',nDec,...
    'NumberOfChannels',nChs,...
    'PolyPhaseOrder', nOrd,...
    'NumberOfVanishingMoments',nVm);

angs = get(nsolt,'Angles');
angs = 2*pi*rand(size(angs));
set(nsolt,'Angles',angs);

nItr = 50;
nCoefs = 30000;

for idx = 1:nItr
    analyzer = saivdr.dictionary.nsoltx.NsoltAnalysis2dSystem('LpPuFb2d',nsolt);
    synthesizer = saivdr.dictionary.nsoltx.NsoltSynthesis2dSystem('LpPuFb2d',nsolt);
    
    iht = saivdr.sparserep.IterativeHardThresholding('Synthesizer',synthesizer,'AdjOfSynthesizer',analyzer);
    [~,coefvec,scales] = step(iht,orgImg(:,:,1),nCoefs);
    
    preangs = get(nsolt,'Angles');
    
    options = optimoptions(@fminunc,'Display','iter-detailed','Algorithm','quasi-newton');
    postangs = fminunc(@(xxx) hogehoge(orgImg(:,:,1),nsolt,xxx,coefvec,scales),preangs,options);
    
    set(nsolt,'Angles',postangs);
end

% Conversion of nsolt to new package style
% nsolt = saivdr.dictionary.utility.fcn_upgrade(nsolt);

% Show the atomic images by using a method atmimshow()
hfig1 = figure(1);
atmimshow(nsolt)
set(hfig1,'Name','Atomic images of NSOLT')

% Instantiation of ISTA system object
import saivdr.restoration.ista.IstaImRestoration
lambda    = 0.00185;                      % lambda
ista = IstaImRestoration(...
    'Synthesizer',        synthesizer,... % Synthesizer (Dictionary)
    'AdjOfSynthesizer',   analyzer,...    % Analyzer (Adj. of dictionary)
    'LinearProcess',      blur,...        % Blur process
    'NumberOfTreeLevels', nLevels,...     % # of tree levels of NSOLT
    'Lambda',             lambda);        % Parameter lambda

%% Create a step monitor system object
% ISTA iteratively approaches to the optimum solution. In order to
% observe the intermediate results, the following class can be used:
%
% * saivdr.utility.StepMonitoringSystem

% Parameters for StepMonitoringSystem
isverbose = true;  % Verbose mode
isvisible = true;  % Monitor intermediate results
hfig2 = figure(2); % Figure to show the source, observed and result image
set(hfig2,'Name','ISTA-based Image Restoration')

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
set(ista,'StepMonitor',stepmonitor);

%% Perform ISTA-based image restoration
% STEP method of IstaImRestoration system object, _ista_ , executes
% the ISTA-based image restoration to deblur the observed image.
% As the result, a restored image
%
% $\hat{\mathbf{u}} = \mathbf{D}\hat{\mathbf{y}}$
%
% is obtained.

fprintf('\n ISTA')
resImg = step(ista,obsImg); % STEP method of IstaImRestoration

%% Extract the final evaluation
% The object of StepMonitoringSystem, _stepmonitor_ , stores the
% evaluation values calculated iteratively in ISTA as a vector. The GET
% method of _stepmonitor_  can be used to extract the number of iterations
% and the sequence of PSNRs.

nItr  = get(stepmonitor,'nItr');
psnrs = get(stepmonitor,'PSNRs');
psnr_ista = psnrs(nItr);

%% Perform Wiener filtering
% As a reference, let us show a result of Wiener filter.

% Create a step monitor system object for the PSNR evaluation
stepmonitor = StepMonitoringSystem(...
    'SourceImage',orgImg,...
    'MaxIter', 1,...
    'IsMSE',  false,...
    'IsPSNR', true,...
    'IsSSIM', false,...
    'IsVisible', false,...
    'IsVerbose', isverbose);

% Use the same blur kernel as that applied to the observed image, obsImg
blurKernel = get(blur,'BlurKernel');

% Estimation of noise to signal ratio
nsr = noise_var/var(orgImg(:));

% Wiener filter deconvolution of Image Processing Toolbox
%wnfImg = deconvwnr(obsImg, blurKernel, nsr);

% Evaluation
fprintf('\n Wiener')
psnr_wfdc = step(stepmonitor,wnfImg); % STEP method of StepMonitoringSystem

%% Compare deblurring performances
% In order to compare the deblurring performances between two methods,
% ISTA-based deblurring with NSOLT and Wiener filter, let us show
% the original, observed and two results in one figure together.

hfig3 = figure(3);

% Original image x
subplot(2,2,1)
imshow(orgImg)
title('Original image {\bf u}')

% Observed image u
subplot(2,2,2)
imshow(obsImg)
title('Observed image {\bf x}')

% Result u^ of ISTA
subplot(2,2,3)
imshow(resImg)
title(['{\bf u}\^ by ISTA  : ' num2str(psnr_ista) ' [dB]'])

% Result u^ of Wiener filter
subplot(2,2,4)
imshow(wnfImg)
title(['{\bf u}\^ by Wiener: ' num2str(psnr_wfdc) ' [dB]'])

%% Release notes
% RELEASENOTES.txt contains release notes on *SaivDr Package*.

type('RELEASENOTES.txt')
