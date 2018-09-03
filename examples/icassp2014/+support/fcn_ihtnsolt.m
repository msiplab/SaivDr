function psnr = fcn_ihtnsolt(params)
%FCN_IHTNSOLT 
%
% SVN identifier:
% $Id: fcn_ihtnsolt.m 683 2015-05-29 08:22:13Z sho $
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

%% Parmeter setting
if nargin < 1
    imgNames = { 'goldhill128', 'lena128', 'barbara128', 'baboon128' };
    nChsSet  = { [ 6 2 ], [ 5 3 ], [ 4 4 ] };
    params.dec = [ 2 2 ];
    params.ord = [ 4 4 ];
    params.imgName = imgNames{1};
    params.chs  = nChsSet{3};
    params.nLevels = 7;
    params.nVm = 1;
    params.nCoefsImpl = 2048;
    params.nCoefs = 2048;
    params.isMonitoring = true;
    params.index = 'best';    
end
srcImg = imresize(im2double(support.fcn_load_testimg(params.imgName)),1);

%% Load learned dictionary
filename = support.fcn_filename(params);
if exist(filename,'file') == 2
    S = load(filename,'lppufbs','bstidx');
    lppufbs = S.lppufbs;
    if strcmp(params.index,'best')
        bstidx  = S.bstidx;
        lppufb  = lppufbs{bstidx};
    elseif strcmp(params.index,'end')
        lppufb  = lppufbs{end};
    else
        lppufb  = lppufbs{params.index};
    end
else
    psnr = -Inf;
    return
end

%% Upgrade if needed
lppufb = saivdr.dictionary.utility.fcn_upgrade(lppufb);

%% Iterative Hard Thresholding
import saivdr.dictionary.nsoltx.NsoltFactory
import saivdr.sparserep.IterativeHardThresholding
synthesizer = NsoltFactory.createSynthesis2dSystem(...
    lppufb);
analyzer    = NsoltFactory.createAnalysis2dSystem(...
    lppufb);
ihtnsolt = IterativeHardThresholding(...
    'Synthesizer',synthesizer,...
    'AdjOfSynthesizer',analyzer,...
    'NumberOfTreeLevels',params.nLevels);
import saivdr.utility.StepMonitoringSystem
stepMonitor = StepMonitoringSystem(...
    'SourceImage',srcImg,...
    'IsPSNR',true,...
    'IsVerbose',params.isMonitoring);
set(ihtnsolt,'StepMonitor',stepMonitor);

[~,coefs,scales] = step(ihtnsolt,srcImg,params.nCoefsImpl);

%% Reconstruction
apxImg = step(synthesizer,coefs,scales);

%% Evaluation
nItr  = get(stepMonitor,'nItr');
psnrs = get(stepMonitor,'PSNRs');
psnr  = psnrs(nItr);
if params.isMonitoring
    imshow(apxImg)
    str = sprintf('PSNR: %6.2f [dB]',psnr);
    disp(str)
    title(str)
end
