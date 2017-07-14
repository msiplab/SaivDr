function output = fcn_HSDirLOT_SURELET_denoise(input)
% fcn_HSDIRLOT_SURELET_DENOISE Removes additive Gaussian white noise 
% using SURE-LET
%
%
% SVN identifier:
% $Id: fcn_HSDirLOT_SURELET_denoise.m 749 2015-09-02 07:58:45Z sho $
%
%  Shogo Muramatsu:
%  ''SURE-LET Image Denoising with Multiple DirLOTs,''
%  Proc. of 2012 Picture Coding Symposium (PCS2012), May 2012.
%
% Requirements: MATLAB R2014a
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

sdir = './filters/';
idx = 1;
%
fname{idx} = 'nsgenlot_d2x2_o4+4_v2.mat'; idx = idx+1;
fname{idx} = 'dirlot_d2x2_o4+4_tvm000.00.mat'; idx = idx+1;
fname{idx} = 'dirlot_d2x2_o4+4_tvm030.00.mat'; idx = idx+1;
fname{idx} = 'dirlot_d2x2_o4+4_tvm060.00.mat'; idx = idx+1;
fname{idx} = 'dirlot_d2x2_o4+4_tvm090.00.mat'; idx = idx+1;
fname{idx} = 'dirlot_d2x2_o4+4_tvm120.00.mat'; idx = idx+1;
fname{idx} = 'dirlot_d2x2_o4+4_tvm-30.00.mat';
%
nTrx = length(fname);
u = cell(nTrx,1);
%
parfor idx = 1:nTrx
    S = load([sdir fname{idx}],'lppufb');
    u{idx} = support.fcn_NSGenLOT_SURELET_denoise(input,S.lppufb);
end
output = 0;
for idx = 1:nTrx
    output = output + u{idx};
end
output = output/nTrx;
