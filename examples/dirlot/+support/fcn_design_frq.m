function [cost, lppufb, spec, costCls] = fcn_design_frq(params)
% FCN_DESIGN_FRQ DirLOT design with frequency domain specification
%
% [cost, lppufb, spec] = fcn_design_frq(params) execute optimization process
% for designing DirLOTs. Input 'params' is a structure which contains 
% parameters to specify the DirLOT design. The default values are used 
% when no input is given and as follows:
%
%   params.dec     = [ 2 2 ];                % # of decimation factor
%   params.ord     = [ 2 2 ];                % # of polyphase order
%   params.Display = 'iter';                 % Display mode
%   params.useParallel = 'always';           % Parallel mode 
%   params.plotFcn = @gaplotbestf;           % Plot function for GA
%   params.populationSize = 20;              % Population size for GA
%   params.eliteCount = 2;                   % Elite count for GA
%   params.mutationFcn = @mutationgaussian;  % Mutation function for GA
%   params.generations = 200;                % # of generations for GA
%   params.stallGenLimit = 100;              % Stall genelation limit
%   params.swmus   = true;                   % Flag for swith mus
%   params.nVm     = 2;                      % # of vanishing moments
%   params.phi     = [];                     % TVM angle in degree 
%                                            % (For emplty phi, classical
%                                            %  VM is adopted)
%   params.dt      = [ 1 1 ];                % Directions of triangle
%   params.costCls = 'AmplitudeErrorEnergy'; % Cost class
%   params.spec    = spec;                   % Freq. spec. in 3-D array
%
% Outputs 'cost,' 'lppufb' and 'spec' are an evaluation of cost function,
% a DirLOT as a result of the design and frequency specification in
% DFT domain, respectively. 'lppufb' is obtaied as an object
% of saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem.
%
% SVN identifier:
% $Id: fcn_design_frq.m 683 2015-05-29 08:22:13Z sho $
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
    nPoints = [ 128 128 ];
    ord = [ 2 2 ];
    dec = [ 2 2 ];
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
    nVm = 2;
    phi = [];
    dt  = [ 1 1 ];
    kl  = [ 3 3 ];
    costCls = 'AmplitudeErrorEnergy';
    
    % Frequency domain specification
    import saivdr.dictionary.nsgenlotx.design.SubbandSpecification
    spec = zeros(nPoints(1),nPoints(2),prod(dec));
    sbsp = SubbandSpecification(...
        'OutputMode','AmplitudeSpecification');
    for idx=1:prod(dec)
        [as, sbIdx] = step(sbsp,nPoints,idx,kl);
        spec(:,:,sbIdx) = as;
    end
else
    ord = params.ord;
    dec = [ 2 2 ];
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
    phi = params.phi;
    dt  = params.dt;
    costCls = params.costCls;    
    %
    spec = params.spec;
end

%% Instantiation of target class
import saivdr.dictionary.nsgenlotx.design.NsGenLotDesignerFrq
import saivdr.dictionary.nsgenlotx.NsGenLotFactory
lppufb = NsGenLotFactory.createLpPuFb2dSystem(...
    'DecimationFactor', dec,...
    'PolyPhaseOrder', ord,...
    'NumberOfVanishingMoments',nVm,...
    'TvmAngleInDegree',phi,...
    'OutputMode','AnalysisFilters');
if isa(lppufb,'LpPuFb2dVm2System')
    set(lppufb,'DirectionOfTriangleY',dt(Direction.VERTICAL));
    set(lppufb,'DirectionOfTriangleX',dt(Direction.HORIZONTAL));
elseif isa(lppufb,'LpPuFb2dTvmSystem')
    set(lppufb,'DirectionOfTriangle',dt(1));
end
designer = NsGenLotDesignerFrq(...
    'CostClass',costCls,...
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
