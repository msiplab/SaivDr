function y = fcn_bwomps2(x,Bsep,A,blocksize,Tdata)
%FCN_BWOMPS 
%
% SVN identifier:
% $Id: fcn_bwomps2.m 867 2015-11-24 04:54:56Z sho $
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014, Shogo MURAMATSU
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

blockdata = x(:);
dc = mean(blockdata(:));
blockdata = blockdata - dc;
gamma = omps(Bsep,A,blockdata,dicttsep(Bsep,A,dictsep(Bsep,A,speye(size(A,2)))),...
    Tdata,'messages',0);
y = reshape(dictsep(Bsep,A,gamma),[blocksize blocksize]) + dc;

end

