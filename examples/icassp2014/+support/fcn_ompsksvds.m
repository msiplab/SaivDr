function psnr = fcn_ompsksvds(params)
%FCN_OMPSKSVDS 
%
% SVN identifier:
% $Id: fcn_ompsksvds.m 683 2015-05-29 08:22:13Z sho $
%
% Requirements: MATLAB R2015b
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
% http://msiplab.eng.niigata-u.ac.jp/
%

%% Parameter setting
if nargin < 1
    imgNames = { 'goldhill128', 'lena128' ,'barbara128', 'baboon128' };
    imgName = imgNames{1};
    Tdata = 8;
    TdataImpl = 8;
    Tdict = 6;
    blocksize = 8;
    odctsize  = 13;
    isMonitoring = true;
    useParallel = false;
else
    imgName = params.imgName;
    Tdata = params.Tdata;
    TdataImpl = params.Tdata;
    Tdict = params.Tdict;
    blocksize = params.blocksize;
    odctsize  = params.odctsize;
    isMonitoring = params.isMonitoring;
    useParallel = params.useParallel;
end
srcImg = imresize(im2double(support.fcn_load_testimg(imgName)),1);

%% Load learned dictionary
fprintf('Learned dictionary condition: b%d o%d n%d d%d %s...\n',...
        blocksize,odctsize,Tdata,Tdict,imgName);
fname = sprintf('./results/ksvds_b%d_o%d_n%d_d%d_%s',...
        blocksize,odctsize,Tdata,Tdict,imgName);
S = load(fname,'A','Bsep');
A = S.A;
Bsep = S.Bsep;

%% Block-wise Sparse Batch-OMP 
fbwomps = @(x) support.fcn_bwomps(x.data,Bsep,A,blocksize,TdataImpl);
apxImg = blockproc(srcImg,[blocksize blocksize],fbwomps,...
    'UseParallel',useParallel);

%% Evaluation
mse = norm(srcImg(:)-apxImg(:))^2/numel(srcImg);
psnr = -10*log10(mse);
if isMonitoring
    imshow(apxImg)
    str = sprintf('PSNR: %6.2f [dB]',psnr);
    disp(str)
    title(str)
end

end
