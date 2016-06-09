%MAIN_SEC_4_1_TABLE1 Computational Time of NSOLT Dictionary Update
%
% This script was used for evaluating the computational time of 
% dictionary date wih quasi-Newton methods through numerical and 
% analytical gradient.
%
% The following materials are also reproduced:
%
%  * materials/tab01.tex
%
% SVN identifier:
% $Id: main_pkg_update.m 875 2016-01-20 01:51:38Z sho $
%
% Requirements: MATLAB R2014a
%
%  * Signal Processing Toolbox
%  * Image Processing Toolbox
%
% Recommended:
%
%  * MATLAB Coder
%
% Copyright (c) 2016, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
ntab = 1;

setpath

%% Parameters
nDecs = [ 4 4 ]; % Decimation factor
nOrds = [ 2 2 ]; % Polyphase order
nLevels = 1;     % Tree level
nVm = 1;         % # of vanishing moments
optfcn = @fminunc; % Optimization function
%
imgLrn = 'barbara32';
nSubImgs = 1;  % # of patches
picSize  = 32; % Patch size
srcImgs = cell(nSubImgs,1); 
rng(0,'twister');
for iSubImg = 1:nSubImgs
    subImgName = [imgLrn 'rnd'];
    orgPatch = ...
        im2double(support.fcn_load_testimg2(subImgName));
    srcImgs{iSubImg} = orgPatch;
end
isOptMus = false; % Flag for optimization of sign parameters.
nCoefs = ceil(numel(srcImgs{1})/8); % # of coefficients for sparse Rep.

%%
chSet = { 10, 11, 12 }; % Evaluation set of # of channels
gradObjSet = { 'off', 'on' }; % Flag for switching the usage of gradient
elapsedTime = zeros(length(chSet),length(gradObjSet)); 
for iCh = 1:length(chSet)
    nChs = chSet{iCh}*[ 1 1 ]; % # of channels
    for iGradObj = 1:length(gradObjSet)
        
        gradObj = gradObjSet{iGradObj}; % on or off
        
        %% Options for optimization
        outfcn = @(x,optimValues,state) ofcn(x,optimValues,state);
        options = optimoptions(optfcn,...
            'PlotFcns',@optimplotfval,...
            'GradObj',gradObj,...
            'MaxIter',20,...
            ...'TolX',1e-10,...
            ...'TolFun',1e-10,...
            ...'OutputFcn',outfcn,...
            'Algorithm','quasi-newton');
        
        %% Instantiation of NSOLT update class
        import saivdr.dictionary.nsoltx.design.*
        import saivdr.dictionary.nsoltx.*
        updater = NsoltDictionaryUpdateGaFmin(...
            'SourceImages', srcImgs,... % Tranining patches
            'NumberOfTreeLevels', nLevels,...
            'OptimizationFunction', optfcn,...
            ...%'MaxIterOfHybridFmin', 2,...
            ...%'GenerationFactorForMus', 2,...
            'GradObj',gradObj,...
            'IsOptimizationOfMus',isOptMus);
        
        %% Instantiation of NSOLT
        nsolt = NsoltFactory.createOvsdLpPuFb2dSystem(...
            'DecimationFactor', nDecs, ...
            'NumberOfChannels', nChs,...
            'PolyPhaseOrder', nOrds,...
            'NumberOfVanishingMoments',nVm,...
            'OutputMode','ParameterMatrixSet');
        %rng(0);
        angs = get(nsolt,'Angles');
        angs = angs + 1e-1*randn(size(angs));
        set(nsolt,'Angles',angs);
        
        %% Instantiation of synthesizer and analyzer
        import saivdr.dictionary.nsoltx.*
        synthesizer = NsoltFactory.createSynthesis2dSystem(...
            nsolt,'IsCloneLpPuFb2d',false);
        analyzer = NsoltFactory.createAnalysis2dSystem(...
            nsolt,'IsCloneLpPuFb2d',false);
        
        %% Instantiation of StepMonitoringSystem
        import saivdr.utility.StepMonitoringSystem
        isverbose = true;  % Verbose mode
        isvisible = true;  % Monitor intermediate results
        hfig1 = figure(1); % Figure to show the source, observed and result image
        set(hfig1,'Name','IHT')
        stepMonitor = StepMonitoringSystem(...
            'SourceImage',   srcImgs{1},... % Original image
            'IsMSE',         false,...      % Switch for MSE  evaluation
            'IsPSNR',        true,...       % Switch for PSNR evaluation
            'IsSSIM',        false,...      % Switch for SSIM evaluation
            'IsVerbose',     isverbose,...  % Switch for verbose mode
            'IsVisible',     isvisible,...  % Switch for display intermediate result
            'ImageFigureHandle',hfig1);     % Figure handle
        
        %% Instantiation of Sparse Coder
        import saivdr.sparserep.IterativeHardThresholding
        sparseCoder = IterativeHardThresholding(...
            'Synthesizer',synthesizer,...
            'AdjOfSynthesizer',analyzer,...
            'NumberOfTreeLevels',nLevels,...
            'StepMonitor',stepMonitor);
        
        %% Sparse Coding
        nImgs = length(srcImgs);
        sprsCoefs   = cell(nImgs,1);
        setOfScales = cell(nImgs,1);
        for iImg = 1:nImgs
            [~, sprsCoefs{iImg}, setOfScales{iImg}] = ...
                step(sparseCoder,srcImgs{iImg},nCoefs);
        end
        
        %% Dictionary Update
        set(updater,'IsOptimizationOfMus',isOptMus);
        set(updater,'SparseCoefficients',sprsCoefs);
        set(updater,'SetOfScales',setOfScales);
        
        tic
        [ nsolt, fval, exitflag ] = step(updater,nsolt,options);
        elapsedTime(iCh,iGradObj) = toc;
        %profile off
        %profile viewer
        %profile on
        
        if strcmp(gradObj,'on')
            options_ = optimoptions(options,...
                'DerivativeCheck','on');
            fprintf('nChs = %d+%d\n',nChs(1),nChs(2));
            step(updater,nsolt,options_);
        end
        
    end
end

%% Generate LaTeX file
sw = StringWriter();

sw.addcr('%#! latex muramatsu')
sw.addcr('%')
sw.addcr('% $Id: TableI.tex 445 2015-08-05 14:13:30Z sho $')
sw.addcr('%')
sw.addcr('\begin{table}[tb]')
sw.addcr('\centering')
sw.addcr('\caption{Computational time for dictionary ')
sw.addcr('update with quasi-Newton methods through numerical ')
sw.addcr('and analytical gradient.}')
sw.addcr('\label{tab:I}')
sw.addcr('\medskip {\footnotesize')
sw.addcr('\begin{tabular}{|c||c|c|c|c|}')
sw.addcr('\hline')
sw.add('$\sharp$Chs. & \multicolumn{2}{c|}{Time [s]} & Accel. & Max. \\')
sw.addcr('\cline{2-3}')
sw.add('$p_\textrm{s}+p_\textrm{a}$ & Numeric & Analytic & ratio & ')
sw.addcr('Rel. Err. \\ \hline\hline')
for iCh = 1:length(chSet) % Summerize elapsed times
    nCh = chSet{iCh};
    sw.add(sprintf('%d+%d & ',nCh,nCh))
    goff = elapsedTime(iCh,1);
    gon  = elapsedTime(iCh,2);
    gain = goff/gon;
    sw.add(sprintf('%6.2f & %6.2f & %6.2f & (DerivativeCheck) ',goff,gon,gain))
    sw.addcr('\\ \hline')
end
sw.addcr('\end{tabular}')
sw.addcr('}')
sw.addcr('\end{table}')
sw.add('\endinput')

%%
disp(sw)
write(sw,sprintf('materials/tab%02d.tex',ntab))