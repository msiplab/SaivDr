%% Quickdesign
% A brief introduction to example-based NSOLT design with *SaivDr Package*

%% Read a source image
% In the followings, an image restoration procedure with this package 
% is described. As a preliminary, let us read an RGB picture as 
% the source image.

imgName = 'ibushi64x64';
srcImg = imread('18_ibushi.normal.png');
width  = 64; % Width
height = 64; % Height
px     = 128; % Horizontal position of cropping
py     = 128; % Vertical position of cropping
%orgImg = im2double(srcImg(py+(1:height),px+(1:width),:));
orgImg = cropImg;

%% Design through dictionary learning 
% Parameters for NSOLT
nLevels = 4;     % # of wavelet tree levels (must be 1 when gradObj = 'on') 
nDecs   = [2 2]; % Decimation factor
nChs    = [4 4]; % # of channels
nOrds   = [2 2]; % Polyphase order
nVm     = 1;     % # of vanishing moments
%nVm = 0;

% Design conditions
%trnImgs{1}   = im2double(rgb2gray(orgImg)); 
%trnImgs{1}   = im2double(orgImg(:,:,1)+1i*orgImg(:,:,2));
trnImgs{1} = orgImg;
nIters       = 8;
nCoefs       = numel(trnImgs{1})/8;
optfcn       = @fminunc;
sparseCoding = 'IterativeHardThresholding';
isFixedCoefs = true;
nUnfixedInitSteps = 0;
isRandomInit = false;
stdOfAng     = pi/6;
gradObj      = 'off'; % Available only for a single-level Type-I NSOLT
% 
options = optimoptions(optfcn);
options = optimoptions(options,'Algorithm','quasi-newton');
options = optimoptions(options,'Display','iter-detailed');
options = optimoptions(options,'UseParallel',true);

% Instantiation of designer
import saivdr.dictionary.nsoltx.design.NsoltDictionaryLearning
designer = NsoltDictionaryLearning(...
    'SourceImages',trnImgs,...
    'NumberOfSparseCoefficients',nCoefs,...
    'DecimationFactor',nDecs,...    
    'NumberOfTreeLevels',nLevels,...
    'NumberOfSymmetricChannel',nChs(1),...
    'NumberOfAntisymmetricChannel',nChs(2),...
    'OptimizationFunction',optfcn,...
    'NumbersOfPolyphaseOrder',nOrds,...
    'OrderOfVanishingMoment',nVm,...
    'SparseCoding',sparseCoding,...
    'IsFixedCoefs',isFixedCoefs,...
    'IsRandomInit',isRandomInit,...
    'GradObj',gradObj);
set(designer,'NumberOfUnfixedInitialSteps',nUnfixedInitSteps);
set(designer,'StdOfAngRandomInit',stdOfAng);

% Iteration 
hfig1 = figure(1);
set(hfig1,'Name','Atomic images of NSOLT')
for iter = 1:nIters
    [ nsolt, mse ] = step(designer,options,[]);
    fprintf('MSE (%d) = %g\n',iter,mse)
    % Show the atomic images by using a method atmimshow()
    atmimshow(nsolt)        
    drawnow
end

%%
fileName = sprintf(...
    'results/nsolt_d%dx%d_c%d+%d_o%d+%d_v%d_l%d_n%d_%s.mat',...
    nDecs(1),nDecs(2),nChs(1),nChs(2),...
    nOrds(1),nOrds(2),nVm,nLevels,...
    nCoefs,imgName);
save(fileName,'nsolt','mse');
