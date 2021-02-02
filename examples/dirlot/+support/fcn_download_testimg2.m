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
% $Id: fcn_download_testimg2.m 749 2015-09-02 07:58:45Z sho $
%
% Requirements: MATLAB R2015b
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
% http://msiplab.eng.niigata-u.ac.jp/
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
