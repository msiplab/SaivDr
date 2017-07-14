function output = fcn_Order2BuildingBlockTypeII(...
    input, mtxW, mtxU, ps, pa, nshift ) %#codegen
% FCN_NSOLTX_SUPEXT_TYPE2 
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014-2017, Shogo MURAMATSU
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
persistent h;
if isempty(h)
    h = saivdr.dictionary.nsoltx.mexsrcs.Order2BuildingBlockTypeII();
end
set(h,'NumberOfSymmetricChannels',ps);
set(h,'NumberOfAntisymmetricChannels',pa);
output = step(h, input, mtxW, mtxU, nshift);
end
