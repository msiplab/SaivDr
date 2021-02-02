function fcn_download_img(isVerbose)
% FCN_DOWNLOAD_IMG 
%
% Copyright (c) Shogo MURAMATSU, 2018
% All rights reserved.
%

% Setting default
if nargin < 1
    isVerbose = true;
end

% Downloading sample images
dstdir = './data/';
if exist(dstdir,'dir') == 7
    fnames = {'lena' 'baboon' 'goldhill' 'barbara'};
    for idx = 1:length(fnames)
        fname = [ fnames{idx} '.png' ];
        if exist(fullfile(dstdir,fname),'file') ~= 2
            img = imread(...
                sprintf('http://homepages.cae.wisc.edu/~ece533/images/%s',...
                fname));
            imwrite(img,fullfile(dstdir,fname))
            if isVerbose
                fprintf('Downloaded and saved %s in %s\n',fname,dstdir);
            end
        else
            if isVerbose
                fprintf('%s already exists in %s\n',fname,dstdir);
            end
        end
    end
else
    me = MException('MsipException:NoSuchFolder', ...
        '%s folder does not exist',dstdir);
    throw(me)
end
