% MAIN_LMAX4IMRSTR Pre-calculation of Lipshitz Constant
%
% This script calculaes and saves the maximum eigen value of 
% the gram matrix of pixel-loss system as a linear process.
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
clear all; clc

%% Parameter settings
nDim = [128 128];

%% Instantiation of degradation
import saivdr.degradation.linearprocess.*
idx = 0;

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
