function x = fcn_observation(linproc,p,strpic,strlinproc,nsigma)
% FCN_OBSERVATION Load or create an observed image
%
% x = fcn_ovservation(linproc,p,strpic,strlinproc,nsigma) 
% loads an observed image x processed by linproc with AWGN with 
% std deviation nsigma for input image p, where 'strpic' and 
% 'strlinproc' are used to identify the file name of observed image x.
% If there doesn't exist the target file, an observed image is created 
% and output as x.
%
% SVN identifier:
% $Id: fcn_observation.m 683 2015-05-29 08:22:13Z sho $
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

% Noise variance for AWGN
noise_var = (nsigma/255)^2;

% File name for observed image
xfname = sprintf(...
    './images/observed_%s_%s_ns%06.2f.tif',strpic,strlinproc,nsigma);

% Load or create an observed image
if exist(xfname,'file') == 2
    disp(['Load ' xfname])
    x = im2double(imread(xfname));
else
    import saivdr.degradation.DegradationSystem
    import saivdr.degradation.noiseprocess.AdditiveWhiteGaussianNoiseSystem
    awgn = AdditiveWhiteGaussianNoiseSystem(...
                'Mean',0,...
                'Variance',noise_var);
    dgrd = DegradationSystem(...
        'LinearProcess',linproc,...
        'NoiseProcess',awgn);
    x = step(dgrd,p);
    imwrite(x,xfname);
end
