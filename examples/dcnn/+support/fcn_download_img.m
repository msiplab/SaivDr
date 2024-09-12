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
if exist(dstdir,'dir') ~= 7
    mkdir(dstdir)
end
for idx = 1:24 %length(fnames)
    fname = "kodim"+num2str(idx,'%02d') + ".png";
    if exist(fullfile(dstdir,fname),'file') ~= 2
        img = imread(...
            "https://www.r0k.us/graphics/kodak/kodak/"+fname);
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
disp('See <a href="https://www.r0k.us/graphics/kodak/">Kodak Lossless True Color Image Suite</a>')
