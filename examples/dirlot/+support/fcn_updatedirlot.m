function [cost, lppufb, costCls] = fcn_updatedirlot(params)
% FCN_UPDATEDIRLOT Update the design result of DirLOT
% 
% [cost, lppufb, costCls] = fcn_updatedirlot(params) updates a design result
% of DirLOT. Input 'params' is a structure which contains parameters 
% specified for the DirLOT design. The default values are used when no 
% input is given and as follows:
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
% Outputs 'cost' and 'lppufb' are an evaluation of cost function
% and a DirLOT as a result of the design, respectively.
%
% SVN identifier:
% $Id: fcn_updatedirlot.m 683 2015-05-29 08:22:13Z sho $
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

if nargin < 1
    nPoints = [ 128 128 ];
    kl    = [ 3 3 ];
    params.dec = [ 2 2 ];
    params.ord = [ 2 2 ];
    params.Display     = 'iter';
    params.useParallel = 'always';   
    params.plotFcn = @gaplotbestf;
    params.populationSize = 20;
    params.eliteCount = 2;
    params.mutationFcn = @mutationgaussian;
    params.generations = 200;
    params.stallGenLimit = 100;
    params.swmus = true;
    params.nVm   = 2;
    params.phi   = 0; %[];
    params.dt    = [ 1 1 ];
    params.costCls = 'AmplitudeErrorEnergy';
     
    % Frequency domain specification
    import saivdr.dictionary.nsgenlotx.design.SubbandSpecification
    params.spec = zeros(nPoints(1),nPoints(2),prod(params.dec));
    sbsp = SubbandSpecification(...
        'OutputMode','AmplitudeSpecification');
    for idx=1:prod(params.dec)
        [as, sbIdx] = step(sbsp,nPoints,idx,kl);
        params.spec(:,:,sbIdx) = as;        
    end  
end

%% File name
if params.nVm < 2 || isempty(params.phi)
    filename = sprintf(...
        'results/nsgenlot_d%dx%d_o%d+%d_v%d.mat',...
        params.dec(1),params.dec(2),...
        params.ord(1),params.ord(2),params.nVm);
elseif params.nVm > 1
    filename = sprintf(...
        'results/dirlot_d%dx%d_o%d+%d_tvm%06.2f.mat',...
        params.dec(1),params.dec(2),...
        params.ord(1),params.ord(2),params.phi);
end

% Load data 
if exist(filename,'file')==2
    s = load(filename);
    lppufb = s.lppufb;
    cost = s.cost;
    spec = s.spec;
    costCls = s.costCls;
    if ~strcmp(costCls,params.costCls)
        error('SaivDr:Invalid cost class, %s ~= %s',...
            costCls,params.costCls);
    end
    if strcmp(params.costCls,'AmplitudeErrorEnergy')
        import saivdr.dictionary.nsoltx.design.AmplitudeErrorEnergy
        costObj = AmplitudeErrorEnergy(...
            'AmplitudeSpecs',spec,...
            'EvaluationMode','All');
    elseif strcmp(params.costCls,'PassBandErrorStopBandEnergy')
        import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
        costObj = PassBandErrorStopBandEnergy(...
            'AmplitudeSpecs',spec,...
            'EvaluationMode','All');
    else
        error('SaivDr:Invalid cost class')
    end
    if abs(cost - step(costObj,lppufb))/cost > 1e-10 
        fprintf('Illeagal cost. File will be removed. %f %f \n',...
            cost, step(costObj,lppufb));
        comstr = sprintf('delete %s\n',filename);
        disp(comstr);
        eval(comstr);
       precost = Inf;
    else
       precost = cost; 
    end 
else
    precost = Inf;
end

% Execute optimization
[cost,lppufb,spec,costCls] = support.fcn_design_frq(params);

% Show atoms
figure(1)
atmimshow(lppufb);

% Show Amp. Res.
figure(2)
H = step(lppufb,[],[]);
freqz2(H(:,:,1));

% Save data
if cost < precost
    if strcmp(costCls,'AmplitudeErrorEnergy')
        import saivdr.dictionary.nsoltx.design.AmplitudeErrorEnergy        
        costObj = AmplitudeErrorEnergy(...
            'AmplitudeSpecs',spec,...
            'EvaluationMode','All');
    elseif strcmp(costCls,'PassBandErrorStopBandEnergy')
        import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy        
        costObj = PassBandErrorStopBandEnergy(...
            'AmplitudeSpecs',spec,...
            'EvaluationMode','All');
    else
        error('SaivDr:Invalid cost class')
    end
    if abs(cost-step(costObj,lppufb))/cost > 1e-10
        fprintf('Illeagal cost. File will not be updated. %f %f \n',...
            cost, step(costObj,lppufb));
    else
        save(filename,'lppufb','cost','spec', 'costCls');
        disp('Updated!');
    end
end
