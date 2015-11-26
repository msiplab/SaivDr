%DISP_QUICKDESIGN Display design examples
%
% This script displays some design examples of nsoltx.
% The example was designed by using the script MIAN_QUICKDESIGN.
%
% SVN identifier:
% $Id: disp_design_examples.m 421 2014-09-18 05:47:10Z sho $
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

%% Conditions
imgName = 'peppers128x128';

nDecs   = [ 2 2 ];
nOrds   = [ 4 4 ];
nCoefs  = 2048;
nChs    = [4 4];
nLevels = 4;
nVm     = 1;

%%
fileName = sprintf(...
    'results/nsolt_d%dx%d_c%d+%d_o%d+%d_v%d_l%d_n%d_%s.mat',...
    nDecs(1),nDecs(2),nChs(1),nChs(2),...
    nOrds(1),nOrds(2),nVm,nLevels,...
    nCoefs,imgName);
if exist(fileName,'file')
    display(fileName)
    S = load(fileName,'nsolt');
    nsolt = S.nsolt;
    figure(1)
    clf
    atmimshow(nsolt)
else
    fprintf('%s : no such file\n',fileName)
end
