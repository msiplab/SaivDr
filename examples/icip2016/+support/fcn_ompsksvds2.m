function [ psnrout, apxImg ] = fcn_ompsksvds2(params)
%FCN_OMPSKSVDS2 
%
% SVN identifier:
% $Id: fcn_ompsksvds2.m 867 2015-11-24 04:54:56Z sho $
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014, Shogo MURAMATSU
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

%% Parameter setting
if nargin < 1
    imgNamesLrn = { 'goldhill128', 'lena128' ,'barbara128', 'baboon128' };
    imgNamesApx = { 'goldhill256', 'lena256' ,'barbara256', 'baboon256' };
    imgNameLrn = imgNamesLrn{1};
    imgNameApx = imgNamesApx{1};
    Tdata = 8;
    TdataImpl = 8;
    Tdict = 6;
    blocksize = 8;
    odctsize  = 13;
    isMonitoring = true;
    useParallel = false;
else
    imgNameLrn   = params.imgNameLrn;
    imgNameApx   = params.imgNameApx;
    Tdata = params.Tdata;
    TdataImpl = params.Tdata;
    Tdict = params.Tdict;
    blocksize = params.blocksize;
    odctsize  = params.odctsize;
    isMonitoring = params.isMonitoring;
    useParallel = params.useParallel;
end
srcImg = imresize(im2double(support.fcn_load_testimg2(imgNameApx)),1);

%% Load learned dictionary
filename = sprintf('./results/ksvds_b%d_o%d_n%d_d%d_%s',...
        blocksize,odctsize,Tdata,Tdict,imgNameLrn);
fprintf('Learned dictionary: %s...\n',filename);
S = load(filename,'A','Bsep');
A = S.A;
Bsep = S.Bsep;

%% Block-wise Sparse Batch-OMP 

fbwomps = @(x) support.fcn_bwomps2(x.data,Bsep,A,blocksize,TdataImpl);
apxImg = blockproc(srcImg,[blocksize blocksize],fbwomps,...
    'UseParallel',useParallel);

%% Evaluation
mse = norm(srcImg(:)-apxImg(:))^2/numel(srcImg);
psnrout = -10*log10(mse);
%psnrout = psnr(srcImg(:),apxImg(:));
if isMonitoring
    imshow(apxImg)
    str = sprintf('PSNR: %6.2f [dB]',psnrout);
    disp(str)
    title(str)
end

end
