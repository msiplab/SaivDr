function str = fcn_updatensolt2( params )
% FCN_UPDATENSOLT Update the design result of NSOLT
%
% str = fcn_updatensolt(params) updates a design result of nsoltx.
% Input 'params' is a structure which contains parameters to specify
% the NOLST design. The default values are used when no input is given
% and as follows:
%
%   params.srcImgs{1} = imresize(im2double(imread('cameraman.tif')),[64 64]);
%   params.imgNameLrn = 'cameraman64';      % Name of source image
%   params.dec = [ 2 2 ];                   % # of decimation factors
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
%   params.isVisible    = false;            % Flag for switch visible mode
%   params.isVerbose    = true;             % Flag for switch verbose mode
%   params.isRandomInit = false;            % Flag for random Init.
%   params.nIters = 2;                      % # of iterations
%
% Outputs 'str' is a string which contains a design summary.
%
% SVN identifier:
% $Id: fcn_updatensolt2.m 867 2015-11-24 04:54:56Z sho $
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014, Shogo MURAMATSU
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
    params.dec = [ 2 2 ];
    params.ord = [ 2 2 ];
    params.chs = [ 4 4 ];
    params.srcImgs{1}  = imresize(im2double(imread('cameraman.tif')),[64 64]);
    params.imgNameLrn  = 'cameraman64';
    params.nCoefs = 256;
    params.nLevels = 1;
    params.Display = 'off';
    params.plotFcn = @optimplotfval;
%     params.populationSize = 20;
%     params.eliteCount = 2;
%     params.mutationFcn = @mutationgaussian;
%     params.generations = 20;
%     params.stallGenLimit = 10;
%     params.maxIterOfHybridFmin = 10;
    params.generationFactorForMus = 2;
    params.sparseCoding = 'IterativeHardThresholding';
    %
    params.optfcn = 'fminsgd';
    params.useParallel = 'never';
    params.isOptMus = true;
    params.isFixedCoefs = true;
    params.nVm = 1;
    params.isVisible = false;
    params.isVerbose = true;
    params.isRandomInit = false;
    params.nUnfixedInitSteps = 0;
    params.stdOfAngRandomInit = pi/6;
    params.nIters = 2;
end

% File name
filename = support.fcn_filename2(params);

% Load data
if exist(filename,'file')==2
    S = load(filename,'mses');
    mses    = S.mses;
    premse  = min(cell2mat(mses));
    prepsnr = -10*log10(premse);
else
    premse  =  Inf;
    prepsnr = -Inf;
end

% Create tasks
[mses,lppufbs] = support.fcn_nsoltdiclrn2(params); %#ok

% Save data
minmse = min(cell2mat(mses)); 
if minmse < premse
    peak = 1; %#ok
    maxpsnr = -10*log10(minmse);
    save(filename,'mses','lppufbs','params');
    str = sprintf('%s Updated! \n\t %6.2f [dB] -> %6.2f [dB]\n', ...
        filename, prepsnr, maxpsnr);
else
    str = sprintf('%s was not updated!',filename);
end
if params.isVerbose
    disp(str)
end
end
