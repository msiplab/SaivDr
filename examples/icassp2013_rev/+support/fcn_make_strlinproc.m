function str = fcn_make_strlinproc(linproc)
% FCN_MAKE_STRLINPROC Generate a string for linear process identification
%
% str = fcn_make_strlinproc(linproc) creates a string of a given
% linear process as input 'linproc' which is an object of 
% saivdr.degradation.linearprocess.AbstLinearProcessSystem.
% 
% SVN identifier:
% $Id: fcn_make_strlinproc.m 683 2015-05-29 08:22:13Z sho $
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

if isa(linproc,'saivdr.degradation.linearprocess.PixelLossSystem')
    losstype = get(linproc,'LossType');
    if strcmp(losstype,'Random')
        density = get(linproc,'Density');
        seed    = get(linproc,'Seed');
        str = sprintf('pls_random_d%3.1f_sd%d',density,seed);
    else
        str = [];
        warning('Not supported yet.');
    end
else
    str = [];
    warning('Not supported yet.');
end
