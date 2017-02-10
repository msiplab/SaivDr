function coefs = fcn_CnsoltAtomConcatenator3d( coefs, scale, pmcoefs, ...
    nch, ord, fpe ) %#codegen
% FCN_NSOLTX_SUPEXT_TYPE1
%    
% SVN identifier: 
% $Id: fcn_NsoltAtomConcatenator3d.m 683 2015-05-29 08:22:13Z sho $
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
persistent h;
if isempty(h)
    h = saivdr.dictionary.nsoltx.CnsoltAtomConcatenator3d(...
        'NumberOfSymmetricChannels',nch(1),...
        'NumberOfAntisymmetricChannels',nch(2));
end
set(h,'IsPeriodicExt',fpe);
set(h,'PolyPhaseOrder',ord);
coefs = step(h, coefs, scale, pmcoefs);
end
