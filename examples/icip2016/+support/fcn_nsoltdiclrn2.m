function [mses, lppufbs] = fcn_nsoltdiclrn2(params)
%FCN_NSOLTDICLRN NSOLT design with dictionary learning
%
% [mses, lppufbs] = fcn_nsoltdiclrn(params) execute learning process
% for designing NSOLTs. Input 'params' is a structure which contains 
% parameters to specify the NOLST design. The default values are used 
% when no input is given and as follows:
% 
%   params.srcImgs{1} = imresize(im2double(imread('cameraman.tif')),[64 64]);
%   params.chs = [ 6 2 ];                   % # of channels
%   params.ord = [ 4 4 ];                   % # of polyphase order
%   params.nCoefs = 16;                     % # of coefficients
%   params.nLevels = 6;                     % # of tree levels
%   params.Display = 'off';                 % Display mode
%   params.useParallel = 'never';           % Parallel mode
%   params.plotFcn = @gaplotbestf;          % Plot function for GA
%   params.populationSize = 20;             % Population size for GA
%   params.eliteCount = 2;                  % Elite count for GA
%   params.mutationFcn = @mutationgaussian; % Mutation function for GA
%   params.generations = 20;                % # of genrations for GA
%   params.stallGenLimit = 10;              % Stall generation limit
%   params.maxIterOfHybridFmin = 10;        % Max. Iter. of Hybrid Func.
%   params.generationFactorForMus = 2;      % Geration factor for MUS
%   params.sparseCoding = 'IterativeHardThresholding';
%   params.optfcn = @ga;                    % Options for optimization
%   params.isOptMus = true;                 % Flag for optimization of MUS
%   params.isFixedCoefs = true;             % Flag if fix Coefs. support
%   params.nVm = 1;                         % # of vanishing moments
%   params.isVisible   = true;              % Flag for switch visible mode
%   params.isVerbose   = true;              % Flag for switch verbose mode
%   params.isRandomInit = false;            % Flag for random Init.
%   params.nIters = 2;                      % # of iterations
%
% Output 'mses' is mean squared errors of sparse approximation results 
% with designed NSOLTs. Output 'lppufbs' is the designed NSOLTs.
%

% Parameters
if nargin < 1
    clc
    dec = [ 4 4 ];
    ord = [ 2 2 ];
    chs = [ 12 12 ];
    rng(0,'twister');
    nSubImgs = 64;
    srcImgs = cell(nSubImgs,1);
    for iSubImg = 1:nSubImgs
        subImgName = 'barbara32rnd';
        orgPatch = ...
            im2double(support.fcn_load_testimg2(subImgName));
        srcImgs{iSubImg} = orgPatch;
    end
    nCoefs = 32^2/8;
    nLevels = 1;
    Display = 'off';
    plotFcn = @optimplotfval;
    populationSize = [];
    eliteCount = [];
    mutationFcn = [];
    generations = [];
    stallGenLimit = [];
    maxIterOfHybridFmin    = [];
    generationFactorForMus = 2;
    sparseCoding = 'IterativeHardThresholding';
    %
    optfcn      = 'fminsgd';
    useParallel = 'never';
    isOptMus = true;
    isFixedCoefs = true;
    nVm = 1;    
    isVisible = true;
    isVerbose = true;
    isRandomInit = true;
    stdOfAngRandomInit = 1e-1;
    nIters  = 20;
    maxIter = 128;
    sgdStep = 'Exponential';
    sgdStepStart = 16;
    sgdStepFinal = 1;
    sgdGaAngInit = 'off';
else
    dec = params.dec;
    ord = params.ord;
    chs = params.chs;
    srcImgs = params.srcImgs;
    nCoefs = params.nCoefs;
    nLevels = params.nLevels;
    Display = params.Display;
    plotFcn = params.plotFcn;
    sparseCoding = params.sparseCoding;
    isRandomInit = params.isRandomInit;
    %
    optfcn = params.optfcn;
    if ~ischar(optfcn)
        if strcmp(func2str(optfcn),'ga')
            populationSize = params.populationSize;
            eliteCount = params.eliteCount;
            mutationFcn = params.mutationFcn;
            generations = params.generations;
            stallGenLimit = params.stallGenLimit;
            maxIterOfHybridFmin    = params.maxIterOfHybridFmin;
            generationFactorForMus = params.generationFactorForMus;
            useParallel = params.useParallel;
        elseif isRandomInit
            stdOfAngRandomInit = params.stdOfAngRandomInit;
        end
    elseif isRandomInit
        stdOfAngRandomInit = params.stdOfAngRandomInit;
    end
    isOptMus = params.isOptMus;
    isFixedCoefs = true;
    nVm = params.nVm;
    isVisible = params.isVisible;
    isVerbose = params.isVerbose;
    nIters = params.nIters;
    maxIter = params.maxIter;
    sgdStep = params.sgdStep;
    sgdStepStart = params.sgdStepStart;
    sgdStepFinal = params.sgdStepFinal;
    sgdGaAngInit = params.sgdGaAngInit;
end

%% Instantiation of target class
import saivdr.dictionary.nsoltx.design.NsoltDictionaryLearning
designer = NsoltDictionaryLearning(...
    'SourceImages',srcImgs,...
    'DecimationFactor',dec,...
    'NumberOfSparseCoefficients',nCoefs,...
    'NumberOfTreeLevels',nLevels,...
    'NumberOfSymmetricChannel',chs(1),...
    'NumberOfAntisymmetricChannel',chs(2),...
    'OptimizationFunction',optfcn,...
    'NumbersOfPolyphaseOrder',ord,...
    'OrderOfVanishingMoment',nVm,...
    'SparseCoding',sparseCoding,...
    'IsFixedCoefs',isFixedCoefs,...
    'IsRandomInit',isRandomInit);
%

if ~ischar(optfcn) 
    if strcmp(func2str(optfcn),'ga')
        set(designer,'MaxIterOfHybridFmin',maxIterOfHybridFmin);
        set(designer,'GenerationFactorForMus',generationFactorForMus);
        options = gaoptimset(optfcn);
        options = gaoptimset(options,'Display',Display);
        options = gaoptimset(options,'UseParallel',useParallel);
        options = gaoptimset(options,'PlotFcn',plotFcn);
        options = gaoptimset(options,'PopulationSize',populationSize);
        options = gaoptimset(options,'EliteCount',eliteCount);
        options = gaoptimset(options,'MutationFcn',mutationFcn);
        options = gaoptimset(options,'Generations',generations);
        options = gaoptimset(options,'StallGenLimit',stallGenLimit);
    else
        if isRandomInit
            set(designer,'StdOfAngRandomInit',stdOfAngRandomInit);
        end
        options = optimoptions(optfcn);
        options = optimoptions(options,'Display','iter');
        options = optimoptions(options,'Algorithm','quasi-newton');
    end
else
    if strcmp(optfcn,'fminsgd')
        if isRandomInit
            set(designer,'StdOfAngRandomInit',stdOfAngRandomInit);
        end
        set(designer,'DictionaryUpdater','NsoltDictionaryUpdateSgd');
        set(designer,'GradObj','on');
        set(designer,'SgdStep',sgdStep);
        set(designer,'SgdStepStart',sgdStepStart);
        set(designer,'SgdStepFinal',sgdStepFinal);
        set(designer,'SgdGaAngInit',sgdGaAngInit);
        options = optimset(...
            'Display','iter',...
            'PlotFcn',plotFcn,...
            'MaxIter',maxIter);
    end
end

%
if (isVisible || isVerbose)
    import saivdr.utility.StepMonitoringSystem    
    stepMonitor = StepMonitoringSystem(...
        'SourceImage',srcImgs{1},...
        'EvaluationType','double',...
        'IsMSE',true,...
        'IsPSNR',true);
    set(stepMonitor,'IsVerbose',isVerbose)
    if isVisible
        hfig1 = findobj(get(groot,'Children'),'Name','Sparse Coding');
        if isempty(hfig1)
            hfig1 = figure;
            set(hfig1,'Name','Sparse Coding')
        end
        set(stepMonitor,'IsVisible',true,...
            'ImageFigureHandle',hfig1);
        %
        hfig2 = findobj(get(groot,'Children'),'Name','Atomic Images');
        if isempty(hfig2)
            hfig2 = figure;
            set(hfig2,'Name','Atomic Images')
        end
    end
    set(designer,'StepMonitor',stepMonitor);    
end

%
lppufbs = cell(nIters,1);
mses = cell(nIters,1);
for iter = 1:nIters
    [ lppufbs{iter}, cost ] = step(designer,options,isOptMus);
    mses{iter} = cost;
    if isVerbose
        fprintf('MSE (%d) = %g\n',iter,mses{iter})
    end
    if isVisible
        hfig2 = findobj(get(groot,'Children'),'Name','Atomic Images');        
        figure(hfig2)
        atmimshow(lppufbs{iter})        
        drawnow
    end 
end

%%
