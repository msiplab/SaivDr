%SETPATH Path setup for *SaivDr Package*
%
% SVN identifier:
% $Id: setpath.m 868 2015-11-25 02:33:11Z sho $
%
% Requirements: MATLAB R2013a
%
% Copyright (c) 2014-2015, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%
% LinedIn: https://www.linkedin.com/in/shogo-muramatsu-627b084b
%
isMexCodesAvailable = true;

if  exist('./+saivdr/','dir') == 7
    envname = 'SAIVDR_ROOT';
    if strcmp(getenv(envname),'')
        setenv(envname,pwd)
    end
    addpath(fullfile(getenv(envname),'.'))
    sdirmexcodes = fullfile(getenv(envname),'mexcodes');
    if isMexCodesAvailable
        addpath(sdirmexcodes)
    elseif strfind(path,sdirmexcodes)  %#ok
        rmpath(sdirmexcodes)
    end
else
    error(['Move to the root directory of SaivDr Toolbox ' ...
           'before executing setpath.']);
end
