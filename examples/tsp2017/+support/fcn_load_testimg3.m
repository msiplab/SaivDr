function [img,strpartpic] = fcn_load_testimg3(strpic)
%FCN_LOAD_TESTIMG3 Load a test image
%
% [img,strpartpic] = fcn_load_testimg3(strpic) loads an volume data
% specified  by the input string 'strpic' from the './volumes'.
% The loaded image is substituted into the output 'img'.
% The second output contains a string which is used for identifing
% the first output.
%
% SVN identifier:
% $Id: fcn_load_testimg3.m 850 2015-10-29 21:19:55Z sho $
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

switch strpic
    case { 'mri128x128x24' }
        nDim = [ 128 128 24 ];
        s = load('mri','D','map');
        src = double(permute(s.D,[1 2 4 3]))/88.0;
        py = 0;
        px = 0;
        pz = 0;
    case { 'mri64x64x24' }
        nDim = [ 64 64 24 ];
        s = load('mri','D','map');
        src = double(permute(s.D,[1 2 4 3]))/88.0;
        py = 32;
        px = 32;
        pz = 0;
    case { 'bunny512x512x361' }
        support.fcn_download_testimg3('bunny-ctscan');
        nDim = [ 512 512 361 ];
        src = double(zeros([512 512 361]));
        for iSlice = 1:361
            fid = fopen(sprintf('./volumes/bunny/%d',iSlice),'r');
            src(:,:,iSlice) = ...
                fread(fid,[512 512],'uint16=>double','ieee-be');
            fclose(fid);
        end
        src = permute(src,[2 1 3]);
        src = src*2^(-12);
        py = 0;
        px = 0;
        pz = 0;
    case { 'cthead256x256x113' }
        support.fcn_download_testimg3('CThead');
        nDim = [ 256 256 113 ];
        src = double(zeros([256 256 113]));
        for iSlice = 1:113
            fid = fopen(sprintf('./volumes/CThead.%d',iSlice),'r');
            src(:,:,iSlice) = ...
                fread(fid,[256 256],'int16=>double','ieee-be');
            fclose(fid);
        end
        src = permute(src,[2 1 3]);
        src = src*2^(-12);
        py = 0;
        px = 0;
        pz = 0;
    case { 'mrbrain256x256x109' }
        support.fcn_download_testimg3('MRbrain');
        nDim = [ 256 256 109 ];
        src = double(zeros([256 256 109]));
        for iSlice = 1:109
            fid = fopen(sprintf('./volumes/MRbrain.%d',iSlice),'r');
            src(:,:,iSlice) = ...
                fread(fid,[256 256],'int16=>double','ieee-be');
            fclose(fid);
        end
        src = permute(src,[2 1 3]);
        src = (src-2^(10))*2^(-12);
        py = 0;
        px = 0;
        pz = 0;
    case { 'mrbrain256x256x96' }
        support.fcn_download_testimg3('MRbrain');
        nDim = [ 256 256 96 ];
        src = double(zeros([256 256 109 ]));
        for iSlice = 1:109
            fid = fopen(sprintf('./volumes/MRbrain.%d',iSlice),'r');
            src(:,:,iSlice) = ...
                fread(fid,[256 256],'int16=>double','ieee-be');
            fclose(fid);
        end
        src = permute(src,[2 1 3]);
        src = (src-2^(10))*2^(-12);
        py = 0;
        px = 0;
        pz = 6;
    case { 'mrbrain128x128x96' }
        support.fcn_download_testimg3('MRbrain');
        nDim = [ 128 128 96 ];
        src = double(zeros([256 256 109 ]));
        for iSlice = 1:109
            fid = fopen(sprintf('./volumes/MRbrain.%d',iSlice),'r');
            src(:,:,iSlice) = ...
                fread(fid,[256 256],'int16=>double','ieee-be');
            fclose(fid);
        end
        src = permute(src,[2 1 3]);
        src = (src-2^(10))*2^(-12);
        py = 64;
        px = 64;
        pz = 6;
    case { 'mrbrain192x192x96' }
        support.fcn_download_testimg3('MRbrain');
        nDim = [ 192 192 96 ];
        src = double(zeros([256 256 109 ]));
        for iSlice = 1:109
            fid = fopen(sprintf('./volumes/MRbrain.%d',iSlice),'r');
            src(:,:,iSlice) = ...
                fread(fid,[256 256],'int16=>double','ieee-be');
            fclose(fid);
        end
        src = permute(src,[2 1 3]);
        src = (src-2^(10))*2^(-12);
        py = 32;
        px = 32;
        pz = 6;
    case { 'mrbrain32x32x32rnd' }
        support.fcn_download_testimg3('MRbrain');
        nDim = [32 32 32];
        src = double(zeros([256 256 109 ]));
        for iSlice = 1:109
            fid = fopen(sprintf('./volumes/MRbrain.%d',iSlice),'r');
            src(:,:,iSlice) = ...
                fread(fid,[256 256],'int16=>double','ieee-be');
            fclose(fid);
        end
        src = permute(src,[2 1 3]);
        src = (src-2^(10))*2^(-12);
        py = randi([32 (192+32-32)]);
        px = randi([32 (192+32-32)]);
        pz = randi([6 (96+6-32)]);
    case { 'mrbrain64x64x64' }
        support.fcn_download_testimg3('MRbrain');
        nDim = [ 64 64 64  ];
        src = double(zeros([ 256 256 109 ]));
        for iSlice = 1:109
            fid = fopen(sprintf('./volumes/MRbrain.%d',iSlice),'r');
            src(:,:,iSlice) = ...
                fread(fid,[ 256 256 ],'int16=>double','ieee-be');
            fclose(fid);
        end
        src = permute(src,[2 1 3]);
        src = (src-2^(10))*2^(-12);
        pz = 22;
        py = 96;
        px = 96;
    case { 'mrbrain32x32x32' }
        support.fcn_download_testimg3('MRbrain');
        nDim = [ 32 32 32 ];
        src = double(zeros([ 256 256 109 ]));
        for iSlice = 1:109
            fid = fopen(sprintf('./volumes/MRbrain.%d',iSlice),'r');
            src(:,:,iSlice) = ...
                fread(fid,[ 256 256 ],'int16=>double','ieee-be');
            fclose(fid);
        end
        src = permute(src,[2 1 3]);
        src = (src-2^(10))*2^(-12);
        pz = 22+32;
        py = 96+32;
        px = 96+32;        
    case { 'walnut296x400x352' }
        support.fcn_download_testimg3('walnut');
        nDim = [ 296 400 352 ];
        fid = fopen('./volumes/walnut.raw','r');
        src = fread(fid,400*296*352,'uint16=>double');
        fclose(fid);
        src = reshape(src,[ 400 296 352 ]);
        src = permute(src,[2 1 3]);
        src = src*2^(-15);
        pz = 0;
        py = 0;
        px = 0;
    case { 'walnut64x64x64' }
        support.fcn_download_testimg3('walnut');
        nDim = [ 64 64 64 ];
        fid = fopen('./volumes/walnut.raw','r');
        src = fread(fid,400*296*352,'uint16=>double');
        fclose(fid);
        src = reshape(src,[ 400 296 352 ]);
        src = permute(src,[2 1 3]);
        src = src*2^(-15);
        py = 116;
        px = 168;
        pz = 144;
    otherwise
        disp(strpic)
        error('Not supported ...')
end
img = src(py+1:py+nDim(1),px+1:px+nDim(2),pz+1:pz+nDim(3));
strpartpic = sprintf('%s_y%d_x%d_z%d_%dx%dx%d',...
    strpic,py,px,pz,nDim(1),nDim(2),nDim(3));