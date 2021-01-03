function filename = fcn_filename(params)
%FCN_FILENAME Generate a file name for saving a result
%
% This script generates an identifiable file name for saving a 
% NSOLT dictionary learing result
%
% SVN identifier:
% $Id: fcn_filename.m 683 2015-05-29 08:22:13Z sho $
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
filename = sprintf(...
    'results/nsolt_d%dx%d_c%d+%d_o%d+%d_v%d_l%d_n%d_%s.mat',...
    params.dec(1),params.dec(2),params.chs(1),params.chs(2),...
    params.ord(1),params.ord(2),params.nVm,params.nLevels,...
    params.nCoefs,params.imgName);
end
