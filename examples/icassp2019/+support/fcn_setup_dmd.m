function  fcn_setup_dmd()
% FCN_SETUP_DMD Setup function for DMD code
%
% Dynamic Mode Decomposition
%   http://bookstore.siam.org/ot149/
%
if exist('CODE/CH01_INTRO/DMD.m','file') ~= 2
    disp('Downloading and unzipping CODE.zip...')
    url = 'http://dmdbook.com/CODE.zip';
    disp(url)
    unzip(url,'./CODE')
else
    disp('./CODE already exists.')
    fprintf('See %s\n', ...
        'http://dmdbook.com/');
end
addpath('./CODE/CH01_INTRO')
end

