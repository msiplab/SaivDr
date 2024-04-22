function fcn_download_testimg3(fname)
%FCN_DOWNLOAD_TESTIMG3 Download a test image
%
% fcn_download_testimage(fname) downloads an image specified by the input
% 'fname' from the following site:
%
%    http://graphics.stanford.edu/data/voldata/
%
% The downloaded image is saved under the directory './volumes/'.
%
% SVN identifier:
% $Id: fcn_download_testimg3.m 850 2015-10-29 21:19:55Z sho $
%
% Requirements: MATLAB R2013b
%
% Copyright (c) 2013-2015, Shogo MURAMATSU
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
if strcmp(fname,'walnut') 
    if ~exist(sprintf('./volumes/%s.raw',fname),'file')         
        url = 'http://voreen.uni-muenster.de/?q=system/files/walnut.zip';
        disp(url)
        unzip(url,'./volumes')            
    else
        fprintf('%s already exists in ./volumes\n',fname);
    end
else
    if ~exist(sprintf('./volumes/%s.tar',fname),'file')         
        url = sprintf('http://graphics.stanford.edu/data/voldata/%s.tar.gz',...
            fname);
        disp(url)
        gunzip(url,'./volumes')    
        untar(sprintf('./volumes/%s.tar',fname),'./volumes')
        fprintf('Downloaded and saved %s in ./volumes\n',fname);
    else
        fprintf('%s already exists in ./volumes\n',fname);
    end
end