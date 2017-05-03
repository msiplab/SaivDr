function coefs = fcn_OLpPrFbAtomExtender1dCodeGen(...
    coefs, scale, pmcoefs, nch, ord, fpe ) %#codegen
% FCN_OLPPRFBATOMEXTENDER1DCODEGEN
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2017, Shogo MURAMATSU
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
    h = saivdr.dictionary.olpprfb.OLpPrFbAtomExtender1d();
end
set(h,'NumberOfSymmetricChannels',nch(1));
set(h,'NumberOfAntisymmetricChannels',nch(2));
set(h,'IsPeriodicExt',fpe);
set(h,'PolyPhaseOrder',ord);
coefs = step(h, coefs, scale, pmcoefs);
end
