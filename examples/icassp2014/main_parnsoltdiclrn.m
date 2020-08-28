%MAIN_PARNSOLTDICLRN NSOLT dictionary learning
%
% This script executes NSOLT dictionary learning
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

%% Parameter settings
params.dec = [ 2 2 ];
params.ord = [ 4 4 ];
params.nCoefs = 2048;
params.Display = 'off';
params.plotFcn = @gaplotbestf;
params.populationSize = 16;
params.eliteCount = 2;
params.mutationFcn = @mutationgaussian;
params.generations   = 40;         % 200 for ICASSP2014
params.stallGenLimit = 20;         % 100 for ICASSP2014
params.maxIterOfHybridFmin = 100;  % 800 for ICASSP2014
params.generationFactorForMus = 2; % 5   for ICASSP2014
params.sparseAprx = 'IterativeHardThresholding';
%
params.optfcn = @ga;
params.useParallel = 'never';
params.isOptMus = true;
params.isFixedCoefs = false; % false means Simultaneous Optimization
params.isMonitoring = false;
params.isRandomInit = true;
params.isVisible    = false;
params.isVerbose    = false;
params.nIters = 15;

%% Condition setting
imgSet = { 'goldhill128', 'lena128', 'barbara128', 'baboon128' };   
vmSet = { 1 };
chSet = { [ 4 4 ] }; % { [ 6 2 ], [ 5 3 ], [ 4 4 ] }; for ICASSP2014
lvSet = num2cell(5:-1:1,[ 1 5 ]);

%% Pre-build of MEX files
% for iCh = 1:length(chSet)
%     import saivdr.dictionary.nsoltx.mexsrcs.*
%     chs = chSet{iCh};
%     fcn_autobuild_atomcnc2d([chs(1) chs(2)]);
%     fcn_autobuild_atomext2d([chs(1) chs(2)]);    
%     if chs(1) == chs(2)
%         fcn_autobuild_bb_type1(chs(1));
%     else
%         fcn_autobuild_bb_type2(chs(1),chs(2));
%     end
% end

%%
paramset = cell(length(imgSet)*length(vmSet)*length(chSet)*length(lvSet),1);
iPar = 1;
for iLv = 1:length(lvSet)
    for iCh = 1:length(chSet)
        for iImg = 1:length(imgSet)
            for iVm = 1:length(vmSet)
                paramset{iPar} = params;
                paramset{iPar}.nVm = vmSet{iVm};
                paramset{iPar}.chs = chSet{iCh};
                paramset{iPar}.nLevels = lvSet{iLv};
                paramset{iPar}.imgName = imgSet{iImg};
                paramset{iPar}.srcImgs{1} = ...
                    im2double(support.fcn_load_testimg(imgSet{iImg}));
                iPar = iPar  + 1;
            end
        end
    end
end

%% Dictionary learning
nReps = 4;
nPars = length(paramset);
for iPar = 1:nPars 
    fprintf('Dictionary learning condition: d%dx%d c%d+%d o%d+%d v%d l%d n%d %s...\n',...
        paramset{iPar}.dec(1),paramset{iPar}.dec(2),...
        paramset{iPar}.chs(1),paramset{iPar}.chs(2),...
        paramset{iPar}.ord(1),paramset{iPar}.ord(2),...
        paramset{iPar}.nVm,paramset{iPar}.nLevels,...
        paramset{iPar}.nCoefs,paramset{iPar}.imgName);
end
%
for iRep = 1:nReps
    parfor iPar = 1:nPars 
        str = support.fcn_updatensolt(paramset{iPar});
    end
end

