%MAIN_SEC_V_B_ATMIMG Impluse responses of 3-D NSOLT synthesizer (fig. 10)
%
% This script was used for observing impluse responses of 3-D NSOLT 
% synthesizer shown in Section V.B.
% 
% The following materials are also reproduced:
%
% - tiff/fig10.tif 
% - materials/fig10.tex
%
% SVN identifier:
% $Id: main_sec_v_b_atmimg.m 855 2015-10-31 07:39:17Z sho $
%
% Requirements: MATLAB R2014a
%
%  * Signal Processing Toolbox
%  * Image Processing Toolbox
%
% Recommended:
% 
%  * MATLAB Coder
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
clear params
close all

nfig    = 10; % Figure number
nfigref =  9; % Figure reference number of training image

%setpath

%% Preparation
close all 
set(0,'defaultAxesFontName','AvantGrade')
set(0,'defaultTextFontName','AvantGrade')

%% Conditions
%
strImg  = 'mrbrain32x32x32'; % Name of the image used for design
nChs    = [ 5 5 ];   % # of channels [ ps pa ];
nDecs   = [ 2 2 2 ]; % Downsampling factor [ M0 M1 M2 ]
nOrds   = [ 2 2 2 ]; % Polyphase order [ N0 N1 N2 ]
nIters  = 10;        % # of iterations for dictionary learning

%%
fileName = sprintf(...
    'results/eznsolt_d%d%d%dc%d%do%d%d%d_ndc1_%s.mat',...
    nDecs(1),nDecs(2),nDecs(3),nChs(1),nChs(2),...
    nOrds(1),nOrds(2),nOrds(3),strImg);
if exist(fileName,'file')
    display(fileName)
    s = load(fileName,'h');
    h = s.h;
    f = flip(flip(flip(h,1),2),3);
    %
    fname = sprintf('tiff/fig%02d',nfig);
    support.fcn_atmimshow3(f,fname)
else
    fprintf('%s : no such file\n',fileName)
end

%% Generate LaTeX file
sw = StringWriter();

sw.addcr('%#! latex double')
sw.addcr('%')
sw.addcr('% $Id: main_sec_v_b_atmimg.m 855 2015-10-31 07:39:17Z sho $');
sw.addcr('%')
sw.addcr('\begin{figure}[tb]')
sw.addcr('\centering')
for iCh = 1:sum(nChs)
    id = ('a'-1)+iCh;
    sw.addcr('\\includegraphics[width=16mm]{Fig%02d%c}',nfig,id)
end
sw.add('\caption{Impluse responses of synthesis filters designed through the ')
sw.add('NSOLT framework for the training volume data shown in ')
sw.addcr(sprintf('Fig.~\\ref{fig:%02d}, ',nfigref))
sw.add('where $\mathbf{M}=\mathrm{diag}(M_0,M_1,M_2)=\mathrm{diag}(')
sw.add([ num2str(nDecs(1)) ',' num2str(nDecs(2)) ',' num2str(nDecs(3)) ])
sw.add(')$, ')
sw.add('$P=p_\mathrm{s}+p_\mathrm{a}=')
sw.add([ num2str(nChs(1)) '+' num2str(nChs(2)) ])
sw.add('$, and ')
sw.add('$\bar{\mathbf{n}}=(N_0,N_1,N_2)^T=(')
sw.add([ num2str(nOrds(1)) ',' num2str(nOrds(2)) ',' num2str(nOrds(3)) ])
sw.addcr(')^T$.}')
sw.addcr(sprintf('\\label{fig:%02d}',nfig))
sw.addcr('\end{figure}')
sw.add('\endinput')

%%
disp(sw)
write(sw,sprintf('materials/fig%02d.tex',nfig))