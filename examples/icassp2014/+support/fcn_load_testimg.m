function [img,strpartpic] = fcn_load_testimg(strpic)
%FCN_LOAD_TESTIMG Load a test image
%
% [img,strpartpic] = fcn_load_testimg(strpic) loads an image specified 
% by the input string 'strpic' from the './images'. 
% The loaded image is substituted into the output 'img'. 
% The second output contains a string which is used for identifing 
% the first output.
%
% SVN identifier:
% $Id: fcn_load_testimg.m 683 2015-05-29 08:22:13Z sho $
%
% Requirements: MATLAB R2015b
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
% http://msiplab.eng.niigata-u.ac.jp/
%

switch strpic
    case { 'goldhill' 'lena' 'barbara' 'baboon' }
        nDim = 512*[1 1];
        support.fcn_download_testimg([strpic '.png']);
        src = imread(['./images/' strpic '.png']);
        py = 0;
        px = 0;
    case { 'goldhill128' 'lena128' 'barbara128' }
        nDim = 128*[1 1];
        support.fcn_download_testimg([strpic(1:end-3) '.png']);
        src = imread(['./images/' strpic(1:end-3) '.png']);
        py = 196;
        px = 196;        
    case { 'goldhill256' 'lena256' 'barbara256' }
        nDim = 256*[1 1];
        support.fcn_download_testimg([strpic(1:end-3) '.png']);
        src = imread(['./images/' strpic(1:end-3) '.png']);
        py = 196;
        px = 196;        
    case { 'baboon128' }
        nDim = 128*[1 1];
        support.fcn_download_testimg([strpic(1:end-3) '.png']);
        src = imread(['./images/' strpic(1:end-3) '.png']);
        py = 16;
        px = 128;
    case { 'baboon256' }
        nDim = 256*[1 1];
        support.fcn_download_testimg([strpic(1:end-3) '.png']);
        src = imread(['./images/' strpic(1:end-3) '.png']);
        py = 16;
        px = 128;        
end
img = src(py+1:py+nDim(1),px+1:px+nDim(2));
strpartpic = sprintf('%s_y%d_x%d_%dx%d',strpic,py,px,nDim(1),nDim(2));
