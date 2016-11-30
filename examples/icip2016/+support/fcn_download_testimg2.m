function fcn_download_testimg2(fname)
%FCN_DOWNLOAD_TESTIMG Download a test image
%
% fcn_download_testimage(fname) downloads an image specified by the input
% 'fname' from the following site:
%
%    http://homepages.cae.wisc.edu/~ece533/imaegs/
%
% The downloaded image is saved under the directory './images/'.
%
% SVN identifier:
% $Id: fcn_download_testimg2.m 867 2015-11-24 04:54:56Z sho $
%
% Requirements: MATLAB R2013b
%
% Copyright (c) 2013-2014, Shogo MURAMATSU
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
if ~exist(sprintf('./images/%s',fname),'file')
    img = imread(...
        sprintf('http://homepages.cae.wisc.edu/~ece533/images/%s',...
        fname));
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    imwrite(img,sprintf('./images/%s',fname));
    fprintf('Downloaded and saved %s in ./images\n',fname);
else
    fprintf('%s already exists in ./images\n',fname);
end
