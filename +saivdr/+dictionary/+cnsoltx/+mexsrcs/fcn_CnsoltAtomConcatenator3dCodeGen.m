function coefs = fcn_CnsoltAtomConcatenator3dCodeGen( coefs, scale, pmcoefs, ...
    nch, ord, fpe ) %#codegen
% FCN_NSOLTX_SUPEXT_TYPE1
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
    h = saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d();
end
set(h,'NumberOfChannels',nch);
set(h,'NumberOfHalfChannels',floor(nch/2));
set(h,'IsPeriodicExt',fpe);
set(h,'PolyPhaseOrder',ord);
coefs = step(h, coefs, scale, pmcoefs);
end
