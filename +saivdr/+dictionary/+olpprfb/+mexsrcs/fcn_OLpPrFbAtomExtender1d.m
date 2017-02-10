function coefs = fcn_OLpPrFbAtomExtender1d( coefs, scale, pmcoefs, ...
    nch, ord, fpe ) %#codegen
% FCN_OLPPRFBATOMEXTENDER1D
%
% SVN identifier:
% $Id: fcn_OLpPrFbAtomExtender1d.m 658 2015-03-17 00:47:13Z sho $
%
% Requirements: MATLAB R2013b
%
% Copyright (c) 2015, Shogo MURAMATSU
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
persistent h;
if isempty(h)
    h = saivdr.dictionary.olpprfb.OLpPrFbAtomExtender1d(...
        'NumberOfSymmetricChannels',nch(1),...
        'NumberOfAntisymmetricChannels',nch(2));
end
set(h,'IsPeriodicExt',fpe);
set(h,'PolyPhaseOrder',ord);
coefs = step(h, coefs, scale, pmcoefs);
end
