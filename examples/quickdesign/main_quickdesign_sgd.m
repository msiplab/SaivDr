%% Quickdesign (SGD)
% A brief introduction to example-based NSOLT design with *SaivDr Package*

%% Read a source image
% In the followings, an image restoration procedure with this package 
% is described. As a preliminary, let us read an RGB picture as 
% the source image.

imgName   = 'peppers';
fileExt   = 'png';
srcImg    = imread([imgName '.' fileExt]);
nSubImgs  = 64;  % # of patches
virWidth  = 128; % Virtual width of training images
virHeight = 128; % Virtual height of training images
virLvTrn  = 1;   % Virtual treee level for training images

% Randam extraction of traning image patchs
height = size(srcImg,1);
width  = size(srcImg,2);
%
subImgs = cell(nSubImgs);
rng(0,'twister');
for iSubImg = 1:nSubImgs
    py = randi([0 (height-virHeight)]);
    px = randi([0 (width-virWidth)]);       
    subImgs{iSubImg} = im2double(imresize(...
        rgb2gray(srcImg(py+(1:virHeight),px+(1:virWidth),:)),...
        1/(2^(virLvTrn-1))));
end

%% Design through dictionary learning 
% Parameters for NSOLT
nLevels = 1;     % # of wavelet tree levels (must be 1 when gradObj = 'on') 
nDecs   = [ 2 2 ]; % Decimation factor
nChs    = [ 4 4 ]; % # of channels
nOrds   = [ 4 4 ]; % Polyphase order
nVm     = 1;     % # of vanishing moments

% Design conditions
nIters       = 8;
nCoefs       = numel(subImgs{1})/8; 
virCoefs     = nCoefs * 2^(virLvTrn-1) * 2^(virLvTrn-1);
isFixedCoefs = true; 
dicUpdater   = 'NsoltDictionaryUpdateSgd'; % Stochastic Gradient Decent
sparseCoding = 'IterativeHardThresholding';
plotFcn      = @optimplotfval;
maxIter      = 128;
gradObj      = 'on'; % Available only for a single-level Type-I NSOLT
sgdStep      = 'Exponential'; % or 'Constant' or 'Reciprocal'
sgdStepStart = 16;
sgdStepFinal = 1;
isRandomInit = true;
isVerbose    = true;
stdOfAngRandomInit = 1e-1;

% Instantiation of designer
import saivdr.dictionary.nsoltx.design.NsoltDictionaryLearning
designer = NsoltDictionaryLearning(...
    'SourceImages',subImgs,...
    'NumberOfSparseCoefficients',nCoefs,...
    'DecimationFactor',nDecs,...    
    'NumberOfLevels',nLevels,...
    'NumberOfSymmetricChannel',nChs(1),...
    'NumberOfAntisymmetricChannel',nChs(2),...
    'NumbersOfPolyphaseOrder',nOrds,...
    'OrderOfVanishingMoment',nVm,...
    'SparseCoding',sparseCoding,...
    'DictionaryUpdater', dicUpdater,...
    'IsFixedCoefs', isFixedCoefs,...
    'IsRandomInit', isRandomInit,...
    'IsVerbose', isVerbose,...
    'StdOfAngRandomInit', stdOfAngRandomInit,...
    'GradObj',gradObj,...
    'SgdStep',sgdStep,...
    'SgdStepStart',sgdStepStart,...
    'SgdStepFinal',sgdStepFinal);
%
options = optimset(...
    'Display','iter',...
    'PlotFcn',plotFcn,...
    'MaxIter',maxIter);

% Iteration 
hfig1 = figure(1);
set(hfig1,'Name','Atomic images of NSOLT')
for iter = 1:nIters
    [ nsolt, mse ] = step(designer,options,[]);
    fprintf('MSE (%d) = %g\n',iter,mse)
    % Show the atomic images by using a method atmimshow()
    figure(hfig1) 
    atmimshow(nsolt)        
    drawnow
end

%%
fileName = sprintf(...
    'results/nsolt_d%dx%d_c%d+%d_o%d+%d_v%d_vl%d_vn%d_%s_sgd.mat',...
    nDecs(1),nDecs(2),nChs(1),nChs(2),...
    nOrds(1),nOrds(2),nVm,virLvTrn,...
    virCoefs,[imgName num2str(virHeight) 'x' num2str(virWidth)]);
save(fileName,'nsolt','mse');
