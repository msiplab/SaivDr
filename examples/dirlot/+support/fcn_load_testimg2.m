function [img,strpartpic] = fcn_load_testimg2(strpic)
%FCN_LOAD_TESTIMG Load a test image
%
% [img,strpartpic] = fcn_load_testimg(strpic) loads an image specified 
% by the input string 'strpic' from the './images'. 
% The loaded image is substituted into the output 'img'. 
% The second output contains a string which is used for identifing 
% the first output.
%
% SVN identifier:
% $Id: fcn_load_testimg2.m 749 2015-09-02 07:58:45Z sho $
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

switch strpic
    case { 'goldhill' 'lena' 'barbara' 'baboon' }
        nDim = 512*[1 1];
        support.fcn_download_testimg2([strpic '.png']);
        src = imread(['./images/' strpic '.png']);
        py = 0;
        px = 0;
    case { 'goldhill512' 'lena512' 'barbara512' 'baboon512' }
        nDim = 512*[1 1];
        support.fcn_download_testimg2([strpic(1:end-3) '.png']);
        src = imread(['./images/' strpic(1:end-3) '.png']);
        py = 0;
        px = 0;
    case { 'goldhill128' 'lena128' 'barbara128' }
        nDim = 128*[1 1];
        support.fcn_download_testimg2([strpic(1:end-3) '.png']);
        src = imread(['./images/' strpic(1:end-3) '.png']);
        py = 196;
        px = 196;        
    case { 'goldhill128rnd' 'lena128rnd' 'barbara128rnd' 'baboon128rnd' }
        nDim = 128*[1 1];
        support.fcn_download_testimg2([strpic(1:end-6) '.png']);
        src = imread(['./images/' strpic(1:end-6) '.png']);
        py = randi([0 (511-128)]);
        px = randi([0 (511-128)]);
    case { 'goldhill64rnd' 'lena64rnd' 'barbara64rnd' 'baboon64rnd' }
        nDim = 64*[1 1];
        support.fcn_download_testimg2([strpic(1:end-5) '.png']);
        src = imread(['./images/' strpic(1:end-5) '.png']);
        py = randi([0 (511-64)]);
        px = randi([0 (511-64)]);        
    case { 'goldhill256' 'lena256' 'barbara256' }
        nDim = 256*[1 1];
        support.fcn_download_testimg2([strpic(1:end-3) '.png']);
        src = imread(['./images/' strpic(1:end-3) '.png']);
        py = 196;
        px = 196;        
    case { 'baboon128' }
        nDim = 128*[1 1];
        support.fcn_download_testimg2([strpic(1:end-3) '.png']);
        src = imread(['./images/' strpic(1:end-3) '.png']);
        py = 16;
        px = 128;
    case { 'baboon256' }
        nDim = 256*[1 1];
        support.fcn_download_testimg2([strpic(1:end-3) '.png']);
        src = imread(['./images/' strpic(1:end-3) '.png']);
        py = 16;
        px = 128;        
    case { 'barbara64' }
        nDim = 64*[1 1];
        support.fcn_download_testimg2([strpic(1:end-2) '.png']);
        src = imread(['./images/' strpic(1:end-2) '.png']);
        py = 256;%320;
        px = 256;                
    case { 'barbara32' }
        nDim = 32*[1 1];
        support.fcn_download_testimg2([strpic(1:end-2) '.png']);
        src = imread(['./images/' strpic(1:end-2) '.png']);
        py = 320;
        px = 256;                        
end
img = src(py+1:py+nDim(1),px+1:px+nDim(2));
strpartpic = sprintf('%s_y%d_x%d_%dx%d',strpic,py,px,nDim(1),nDim(2));
