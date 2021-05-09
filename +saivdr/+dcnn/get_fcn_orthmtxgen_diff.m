function [fcnhandler,flag] = get_fcn_orthmtxgen_diff(angles)
%GET_FCN_ORTHMTXGEN_DIFF
%
% Requirements: MATLAB R2021a
%
% Copyright (c) 2020-2021, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
import saivdr.dcnn.mexsrcs.*
fcnname = 'fcn_orthmtxgen_diff';
%
%
datatype = underlyingType(angles);
if isgpuarray(angles)
    device = 'gpu';
else
    device = 'cpu';
end
%
if nargin < 1
    mexname = '';
else
    mexname = [fcnname '_' datatype '_on_' device '_mex'];
end
ftypemex = exist(mexname, 'file');
if ftypemex == 3 % MEX file exists
    fcnhandler = str2func(mexname);
    flag       = true;
else
    fcnhandler = str2func(fcnname);
    flag       = false;
end
end