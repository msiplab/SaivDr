function [ psnr, apxImg ] = fcn_ihtnsolt2(params)
%FCN_IHTNSOLT 
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014-2020, Shogo MURAMATSU
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
    params.imgNameLrn = imgNames{1};
    params.chs  = nChsSet{3};
    params.nLevels = 7;
    params.nLevelsImpl = 7;
    params.nVm = 1;
    params.nCoefsImpl = 2048;
    params.nCoefs = 2048;
    params.isMonitoring = true;
    params.index = 'best';    
    params.filterDomain = 'frequency';
end
srcImg = im2double(support.fcn_load_testimg2(params.imgNameApx));

%% Load learned dictionary
filename = support.fcn_filename2(params);
if exist(filename,'file') == 2
    S = load(filename,'lppufbs');
    lppufbs = S.lppufbs;
    if strcmp(params.index,'best')
        S = load(filename,'mses');
        mses_ = cell2mat(S.mses);
        [~,bstidx] = min(mses_(:));
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
import saivdr.sparserep.IterativeHardThresholding
if strcmp(params.filterDomain,'frequency')
    import saivdr.dictionary.generalfb.Analysis2dSystem
    import saivdr.dictionary.generalfb.Synthesis2dSystem
    %
    release(lppufb)
    set(lppufb,'OutputMode','AnalysisFilters')
    analysisFilters = step(lppufb,[],[]);
    %
    release(lppufb)
    set(lppufb,'OutputMode','SynthesisFilters')
    synthesisFilters = step(lppufb,[],[]);
    %
    analyzer    = Analysis2dSystem(...
        'DecimationFactor',params.dec,...
        'AnalysisFilters',analysisFilters,...
        'FilterDomain','Frequency');
    synthesizer = Synthesis2dSystem(...
        'DecimationFactor',params.dec,...
        'SynthesisFilters',synthesisFilters,...
        'FilterDomain','Frequency');
    setFrameBound(synthesizer,1);
elseif strcmp(params.filterDomain,'lattice_termination') 
    import saivdr.dictionary.nsoltx.NsoltFactory
    synthesizer = NsoltFactory.createSynthesis2dSystem(...
        lppufb,...
        'BoundaryOperation','Termination');
    analyzer    = NsoltFactory.createAnalysis2dSystem(...
            lppufb,...
        'BoundaryOperation','Termination');
elseif strcmp(params.filterDomain,'lattice_periodic') 
    import saivdr.dictionary.nsoltx.NsoltFactory
    synthesizer = NsoltFactory.createSynthesis2dSystem(...
        lppufb,...
        'BoundaryOperation','Termination');
    analyzer    = NsoltFactory.createAnalysis2dSystem(...
            lppufb,...
        'BoundaryOperation','Termination');
else
    error('Unsupported FilterDomain: %s',params.filterDomain)
end
%
ihtnsolt = IterativeHardThresholding(...
    'Synthesizer',synthesizer,...
    'AdjOfSynthesizer',analyzer,...
    'NumberOfLevels',params.nLevelsImpl);
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
