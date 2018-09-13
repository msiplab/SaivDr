function [mses, lppufbs] = fcn_nsoltdiclrn(params)
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
% SVN identifier:
% $Id: fcn_nsoltdiclrn.m 683 2015-05-29 08:22:13Z sho $
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
  
% Parameters
if nargin < 1
    ord = [ 4 4 ];
    chs = [ 4 4 ];
    srcImgs{1}  = imresize(im2double(imread('cameraman.tif')),[64 64]);
    nCoefs = 16;
    nLevels = 6;
    Display = 'off';
    plotFcn = @gaplotbestf;
    populationSize = 20;
    eliteCount = 2;    
    mutationFcn = @mutationgaussian;
    generations = 20;
    stallGenLimit = 10;
    maxIterOfHybridFmin    = 10;
    generationFactorForMus = 2;
    sparseCoding = 'IterativeHardThresholding';
    %
    optfcn = @ga;
    useParallel = 'never';
    isOptMus = true;
    isFixedCoefs = true;
    nVm = 1;    
    isVisible = true;
    isVerbose = true;
    isRandomInit = false;
    nIters = 2;
else
    ord = params.ord;
    chs = params.chs;
    srcImgs = params.srcImgs;
    nCoefs = params.nCoefs;
    nLevels = params.nLevels;
    Display = params.Display;
    plotFcn = params.plotFcn;
    populationSize = params.populationSize;
    eliteCount = params.eliteCount;
    mutationFcn = params.mutationFcn;
    generations = params.generations;
    stallGenLimit = params.stallGenLimit;
    maxIterOfHybridFmin    = params.maxIterOfHybridFmin;
    generationFactorForMus = params.generationFactorForMus;
    sparseCoding = params.sparseCoding;
    %
    optfcn = params.optfcn;
    useParallel = params.useParallel;
    isOptMus = params.isOptMus;
    isFixedCoefs = true;
    nVm = params.nVm;
    isVisible = params.isVisible;
    isVerbose = params.isVerbose;
    isRandomInit = params.isRandomInit;
    nIters = params.nIters;
end

%% Instantiation of target class
import saivdr.dictionary.nsoltx.design.NsoltDictionaryLearning
designer = NsoltDictionaryLearning(...
    'SourceImages',srcImgs,...
    'NumberOfSparseCoefficients',nCoefs,...
    'NumberOfLevels',nLevels,...
    'NumberOfSymmetricChannel',chs(1),...
    'NumberOfAntisymmetricChannel',chs(2),...
    'OptimizationFunction',optfcn,...
    'NumbersOfPolyphaseOrder',ord,...
    'OrderOfVanishingMoment',nVm,...
    'MaxIterOfHybridFmin',maxIterOfHybridFmin,...
    'GenerationFactorForMus',generationFactorForMus,...
    'SparseCoding',sparseCoding,...
    'IsFixedCoefs',isFixedCoefs,...
    'IsRandomInit',isRandomInit);
%

if strcmp(func2str(optfcn),'ga')
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
    options = optimoptions(optfcn);
    options = optimoptions(options,'Display','iter');
end

%
if (isVisible || isVerbose)
    import saivdr.utility.StepMonitoringSystem    
    stepMonitor = StepMonitoringSystem(...
        'SourceImage',srcImgs{1},...
        'EvaluationType','uint8',...
        'IsMSE',true,...
        'IsPSNR',true);
    set(stepMonitor,'IsVerbose',isVerbose)
    if isVisible
        hfig1 = figure(1);
        set(stepMonitor,'IsVisible',true,...
            'ImageFigureHandle',hfig1);
        hfig2 = figure(2);
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
        figure(hfig2)
        atmimshow(lppufbs{iter})        
    end 
end
