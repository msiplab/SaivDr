function [cost, lppufb] = fcn_updatensolt(params)
% FCN_UPDATENSOLT Update the design result of NSOLT
% 
% [cost, lppufb] = fcn_updatensolt(params) updates a design result
% of nsoltx. Input 'params' is a structure which contains parameters 
% specified for the NOLST design. The default values are used when no 
% input is given and as follows:
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
% Outputs 'cost' and 'lppufb' are an evaluation of cost function
% and an NSOLT as a result of the design, respectively.
%
% SVN identifier:
% $Id: fcn_updatensolt.m 683 2015-05-29 08:22:13Z sho $
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

if nargin < 1
    params.nPoints = [128 128];
    params.dec = [2 2];
    params.chs = [5 2];
    params.ord = [ 2 2 ];
    params.Display = 'iter';
    params.useParallel = 'always';   
    params.plotFcn = @gaplotbestf;
    params.populationSize = 20;
    params.eliteCount = 2;
    params.mutationFcn = @mutationgaussian;
    params.generations = 200;
    params.stallGenLimit = 100;
    params.swmus = true;
    params.nVm   = 1;
    
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
    params.spec = specPassStopBand;    
end

%% File name
filename = sprintf(...
    'results/nsolt_d%dx%d_c%d+%d_o%d+%d_v%d.mat',...
    params.dec(1),params.dec(2),params.chs(1),params.chs(2),...
    params.ord(1),params.ord(2),params.nVm);

% Load data 
if exist(filename,'file')==2
    s = load(filename);
    lppufb = saivdr.dictionary.utility.fcn_upgrade(s.lppufb);
    cost = s.cost;
    spec = s.spec;
    import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
    psbe = PassBandErrorStopBandEnergy('AmplitudeSpecs',spec);
    if abs(cost - step(psbe,lppufb))/cost > 1e-10 
        fprintf('Illeagal cost. File will be removed. %f %f \n',...
            cost, step(psbe,lppufb));
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
[cost,lppufb,spec] = support.fcn_design_frq(params);

% Show atoms
figure(1)
atmimshow(lppufb);

% Show Amp. Res.
figure(2)
H = step(lppufb,[],[]);
freqz2(H(:,:,1));

% Save data
if cost < precost
    import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
    psbe = PassBandErrorStopBandEnergy(...
        'AmplitudeSpecs',spec,...
        'EvaluationMode','All');
    if abs(cost-step(psbe,lppufb))/cost > 1e-10
        fprintf('Illeagal cost. File will not be updated. %f %f \n',...
            cost, step(psbe,lppufb));
    else
        save(filename,'lppufb','cost','spec');
        disp('Updated!');
    end
end
