% MAIN_DIRLOT_RAMPROT Script for evaluationg TVM characteristics of DirLOT
%
% SVN identifier:
% $Id: main_dirlot_ramprot.m 683 2015-05-29 08:22:13Z sho $
%
% Shogo Muramatsu, Dandan Han, Tomoya Kobayashi and Hisakazu Kikuchi: 
%  ''Directional Lapped Orthogonal Transform: Theory and Design,'' 
%  IEEE Trans. on Image Processing, Vol.21, No.5, pp.2434-2448, May 2012.
%  (DOI: 10.1109/TIP.2011.2182055)
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
params.dec = 2;  % Decimation factor
params.ord = 4;  % Polyphase order
params.phi = 60; % { [], -30, 30, 60, 120 }

params.eps = 1e-15;
params.dim = [ 32 32 ];
transition = 0.25;   % Transition band width
params.location = './filters';

[dirlotNorm0,dctNorm0,phixd,arrayCard] = support.fcn_dirlot_ramprot(params);

figure(1)
plot(phixd,dirlotNorm0/arrayCard,phixd,dctNorm0/arrayCard)
ylabel('Sparsity')
xlabel('TVM angle [deg]')

save results/dirlot_ramprot_results phixd dirlotNorm0 dctNorm0 arrayCard
