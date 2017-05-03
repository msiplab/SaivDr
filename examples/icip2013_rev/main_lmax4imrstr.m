% MAIN_LMAX4IMRSTR Pre-calculation of Lipshitz Constant
%
% Requirements: MATLAB R2013b
%
% Copyright (c) 2013-2016, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
clc

%% Parameter settings
nDim = [128 128];

%% Instantiation of degradation
import saivdr.degradation.linearprocess.*
idx = 0;
% Super-resolution
idx = idx+1;
dfactor = 2;
sigma = 2;
shblur = 'Gaussian';
fname_dec = sprintf('./lmax/dec_gaussian_d%d_s%3.1f_%dx%d.mat',...
    dfactor,sigma,nDim(1),nDim(2));
dgrd{idx} = DecimationSystem(...
    'VerticalDecimationFactor',dfactor,...
    'HorizontalDecimationFactor',dfactor,...
    'BlurType',shblur,...
    'SigmaOfGaussianKernel',sigma,...
    'UseFileForLambdaMax',true,...
    'FileNameForLambdaMax',fname_dec);
fname{idx} = fname_dec;

% Deblurring
idx = idx+1;
sigma = 2;
shblur = 'Gaussian';
fname_blr = sprintf('./lmax/blr_gaussian_s%3.1f_%dx%d.mat',...
    sigma,nDim(1),nDim(2));
dgrd{idx} = BlurSystem(...
    'BlurType',shblur,...
    'SigmaOfGaussianKernel',sigma,...
    'UseFileForLambdaMax',true,...
    'FileNameForLambdaMax',fname_blr);
fname{idx} = fname_blr;

% Inpainting
idx = idx+1;
losstype = 'Random';
density = 0.2;
seed = 0;
fname_pls = sprintf('./lmax/pls_random_d%3.1f_sd%d_%dx%d.mat',...
    density,seed,nDim(1),nDim(2));
dgrd{idx} = PixelLossSystem(...
    'LossType',losstype,...
    'Density',density,...
    'Seed',seed,...
    'UseFileForLambdaMax',true,...
    'FileNameForLambdaMax',fname_pls);
fname{idx} = fname_pls;

%% Run pre-calculation of Lipschitz constants
for idx = 1:length(dgrd)
    step(dgrd{idx},ones(nDim));
    valueL = get(dgrd{idx},'LambdaMax');    
    fprintf('%s : %6.3f\n',fname{idx},valueL)
end
