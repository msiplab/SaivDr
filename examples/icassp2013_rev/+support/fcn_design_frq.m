function [cost, lppufb, spec] = fcn_design_frq(params)
% FCN_DESIGN_FRQ NSOLT design with frequency domain specification
%
% [cost, lppufb] = fcn_design_frq(params) execute optimization process
% for designing NSOLTs. Input 'params' is a structure which contains 
% parameters to specify the NOLST design. The default values are used 
% when no input is given and as follows:
%
%   params.nPoints = [128 128];              % # of DFT points
%   params.dec = [ 2 2 ];                    % # of decimation factor
%   params.chs = [ 5 2 ];                    % # of channels
%   params.ord = [ 2 2 ];                    % # of polyphase order
%   params.Display = 'iter';                 % Display mode
%   params.useParallel = 'always';           % Parallel mode 
%   params.plotFcn = @gaplotbestf;           % Plot function for GA
%   params.populationSize = 20;              % Population size for GA
%   params.eliteCount = 2;                   % Elite count for GA
%   params.mutationFcn = @mutationgaussian;  % Mutation function for GA
%   params.generations = 200;                % # of generations for GA
%   params.stallGenLimit = 100;              % Stall genelation limit
%   params.swmus = true;                     % Flag for swith mus
%   params.nVm   = 1;                        % # of vanishing moments
%   params.spec = specPassStopBand;          % Freq. spec. in 3-D array
%
% Outputs 'cost,' 'lppufb' and 'spec' are an evaluation of cost function,
% an NSOLT as a result of the design and frequency specification in
% DFT domain, respectively. 'lppufb' is obtaied as an object
% of saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem.
%
% SVN identifier:
% $Id: fcn_design_frq.m 683 2015-05-29 08:22:13Z sho $
%
% Requirements: MATLAB R2013b
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
% LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
%

% Parameters
if nargin < 1
    ord = [ 2 2 ];
    dec = [ 2 2 ];
    chs = [ 5 2 ];
    display = 'off';
    useParallel = 'never';
    plotFcn = @gaplotbestf;
    populationSize = 20;
    eliteCount = 2;
    mutationFcn = @mutationgaussian;
    generations = 200;
    stallGenLimit = 100;
    %
    swmus = false;
    nVm = 1;
    
    % Frequency domain specification
    P =  ones(8); % Pass band value
    S = -ones(8); % Stop band value
    T = zeros(8); % Transition band value
    A = zeros(8);
    for iCol = 1:8
        for iRow = 1:8
            if iCol>(9-iRow)
                A(iRow,iCol) = 0;
            else
                A(iRow,iCol) = 1;
            end
        end
    end
    specPassStopBand(:,:,1) =  [
        S S S S S S S S S S S S S S S S ;
        S S S S S S S S S S S S S S S S ;
        S S S S S S S S S S S S S S S S ;
        S S S T T T T T T T T T T S S S ;
        S S S T T T T T T T T T T S S S ;
        S S S T T P P P P P P T T S S S ;
        S S S T T P P P P P P T T S S S ;
        S S S T T P P P P P P T T S S S ;
        S S S T T P P P P P P T T S S S ;
        S S S T T P P P P P P T T S S S ;
        S S S T T P P P P P P T T S S S ;
        S S S T T T T T T T T T T S S S ;
        S S S T T T T T T T T T T S S S ;
        S S S S S S S S S S S S S S S S ;
        S S S S S S S S S S S S S S S S ;
        S S S S S S S S S S S S S S S S ]; % E
    specPassStopBand(:,:,2) = [
        T T T T T T T T T T T T T T T T ;
        T P P P P P P T T S S S S S S T ;
        T P P P P P P T T S S S S S S T ;
        T T T T T T T T T S S S S S S T ;
        T T T T T T T T T S S S S S S T ;
        S S S S S S S S S S S S S S S S ;
        S S S S S S S S S S S S S S S S ;
        S S S S S S S S S S S S S S S S ;
        S S S S S S S S S S S S S S S S ;
        S S S S S S S S S S S S S S S S ;
        S S S S S S S S S S S S S S S S ;
        T S S S S S S T T T T T T T T T ;
        T S S S S S S T T T T T T T T T ;
        T S S S S S S T T P P P P P P T ;
        T S S S S S S T T P P P P P P T ;
        T T T T T T T T T T T T T T T T ];
    %
    specPassStopBand(:,:,3) = flipud(specPassStopBand(:,:,2)); % E
    specPassStopBand(:,:,4) = specPassStopBand(:,:,2).'; % E
    specPassStopBand(:,:,5) = fliplr(specPassStopBand(:,:,4)); % E
    specPassStopBand(:,:,6) = circshift(specPassStopBand(:,:,1),[0 64]); % O
    specPassStopBand(:,:,7) = specPassStopBand(:,:,6).'; % O
    %
    spec = specPassStopBand;
else
    ord = params.ord;
    dec = [ 2 2 ];
    chs = [ 5 2 ];
    display = params.Display;
    useParallel = params.useParallel;
    plotFcn = params.plotFcn;
    populationSize = params.populationSize;
    eliteCount = params.eliteCount;
    mutationFcn = params.mutationFcn;
    generations = params.generations;
    stallGenLimit = params.stallGenLimit;
    %
    swmus = params.swmus;
    nVm = params.nVm;
    %
    spec = params.spec;
end

%% Instantiation of target class
import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
import saivdr.dictionary.nsoltx.design.NsoltDesignerFrq
import saivdr.dictionary.nsoltx.NsoltFactory
lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
    'DecimationFactor', dec,...
    'NumberOfChannels', chs,...
    'PolyPhaseOrder', ord,...
    'NumberOfVanishingMoments',nVm,...
    'OutputMode','AnalysisFilters');
designer = NsoltDesignerFrq(...
    'AmplitudeSpecs',spec,...
    'OptimizationFunction',@ga);

% Parameter setup
angles = get(lppufb,'Angles');
popInitRange = [angles(:)-pi angles(:)+pi].';
%
if swmus
    mus = get(lppufb,'Mus');
    mus = 2*round(rand(size(mus))) - 1;
    set(lppufb,'Mus',mus);
end
%
options = gaoptimset('ga');
options = gaoptimset(options,'Display',display);
options = gaoptimset(options,'UseParallel',useParallel);
options = gaoptimset(options,'PlotFcn',plotFcn);
options = gaoptimset(options,'PopulationSize',populationSize);
options = gaoptimset(options,'EliteCount',eliteCount);
options = gaoptimset(options,'MutationFcn',mutationFcn);
options = gaoptimset(options,'PopInitRange',popInitRange);
options = gaoptimset(options,'Generations',generations);
options = gaoptimset(options,'StallGenLimit',stallGenLimit);

% First optimization
if swmus
    set(designer,'IsOptimizationOfMus',false);
else
    set(designer,'IsOptimizationOfMus',true);
end
[lppufb, cost0, ~] = step(designer,lppufb,options);
lppufb0 = clone(lppufb);
disp(cost0)

% Second optimization
set(designer,'IsOptimizationOfMus',true);
[lppufb, cost1, ~] = step(designer,lppufb,options);
lppufb1 = clone(lppufb);
disp(cost1)

% Selection
if cost0 < cost1
    lppufb = lppufb0;
    cost = cost0;
else
    lppufb = lppufb1;
    cost = cost1;
end
