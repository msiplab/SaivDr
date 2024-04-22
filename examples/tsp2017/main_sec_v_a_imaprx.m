%MAIN_SEC_V_A_IMAPRX Sparse approximation with 2-D NSOLT (fig. 7)
%
% This script was used for observing sparse approximation results with 2-D NSOLTs
% in Section V.A.
%
% The following materials are also reproduced:
%
%  * tiff/fig07x.tif ( x in {a,b,c,...,l} )
%  * materials/fig07.tex
%
% SVN identifier:
% $Id: main_sec_v_a_imaprx.m 856 2015-11-02 02:01:16Z sho $
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
close all
clear params

nfig    = 7;     % Figure number
nfigref = 6;     % Figure reference number of training image
ntabref = 4;     % Table reference number of PSNR table
isEdit  = false; % Edit mode

%setpath

%% Parameters
nDec = [ 2 2 ]; % M0 M1
maxOrd = 3; 
nChsSp = [ 3 3 ]; % ps pa (Separable OLT)
nChsNs = [ % ps pa (NSOLT) 
    2 2 ; 
    2 3 ; 
    3 2 ; 
    3 3 ];
idx = 1;
% Separable OLT
for iChs=1:size(nChsSp,1)
    for iOrd = 1:maxOrd
        params{idx} = struct(...
            'nDec',nDec,...
            'nChs',nChsSp(iChs,:),...
            'nOrd',iOrd*[1 1],...
            'opt','sep');
        idx = idx+1;
    end
end
% NSOLT
for iChs = 1:size(nChsNs,1)
    for iOrd = 1:maxOrd
        params{idx} = struct(...
            'nDec',nDec,...
            'nChs',nChsNs(iChs,:),...
            'nOrd',iOrd*[1 1],...
            'opt','');
        idx = idx+1;
    end
end
%
strImg   = 'barbara64'; % Training image
sparsity = 1/4;         % Sparsity in ratio

%% Preparation
import saivdr.dictionary.generalfb.Analysis2dSystem
import saivdr.dictionary.generalfb.Synthesis2dSystem
import saivdr.sparserep.IterativeHardThresholding
srcImg = support.fcn_load_testimg2(strImg);
srcImg = im2double(srcImg);
figure(1),imshow(srcImg)

%% Sparsity
nSparseCoefs = round(numel(srcImg)*sparsity);

%% Sparse approximation process
sw = StringWriter();

sw.addcr('%#! latex double')
sw.addcr('%')
sw.addcr('% $Id: main_sec_v_a_imaprx.m 856 2015-11-02 02:01:16Z sho $')
sw.addcr('%')
sw.addcr('\begin{figure}[tb]')
sw.addcr('\centering')
for idx = 1:length(params)
    nDec = params{idx}.nDec;
    nOrd = params{idx}.nOrd;
    ps = params{idx}.nChs(1);
    pa = params{idx}.nChs(2);
    opt = params{idx}.opt;
    %
    if ~isEdit
        s = load(sprintf('results/eznsolt_d%d%dc%d%do%d%d%s_ndc1_%s',...
            nDec(1),nDec(2),ps,pa,nOrd(1),nOrd(2),opt,strImg),'h');
        h = s.h;
        f = flip(flip(h,1),2);
        analyzer    = Analysis2dSystem('DecimationFactor',nDec,...
            'AnalysisFilters',h,...
            'FilterDomain','Frequency');
        synthesizer = Synthesis2dSystem('DecimationFactor',nDec,...
            'SynthesisFilters',f,...
            'FilterDomain','Frequency');
        iht = IterativeHardThresholding(...
            'Synthesizer',synthesizer,...
            'AdjOfSynthesizer',analyzer);
        resImg = step(iht,srcImg,nSparseCoefs);
        ids = idx;
        id = ('a'-1)+ids;
        apxImg = srcImg-resImg;
        figure(2),subplot(ceil(length(params)/3),3,ids),imshow(apxImg)
        fname = sprintf('fig%02d%c',nfig,id);
        imwrite(apxImg,['tiff/' fname '.tif'])
        %
        ids = idx;
        id = ('a'-1)+ids;
        sw.addcr('\begin{minipage}[b]{16mm}')
        sw.addcr('  \centering')
        sw.add('  \centerline{\includegraphics[width=\textwidth]{')
        sw.add(fname)
        sw.addcr('}}')
        sw.addcr(sprintf('  \\centerline{\\footnotesize (%c)}\\medskip',id));
        sw.addcr('\end{minipage}')
        if nOrd(1) < maxOrd && nOrd(2) < maxOrd
            sw.addcr('\hspace{2mm}')
        else
            sw.addcr('')
        end
    end
end
sw.add('\caption{Sparse approximation results with 2-D NSOLTs dictionaries ')
sw.add('through IHT for the training image shown in ')
sw.add(sprintf('Fig.~\\ref{fig:%02d}, ',nfigref))
sw.add('where $\mathbf{M}=\mathrm{diag}(M_0,M_1)=\mathrm{diag}(')
sw.add([ num2str(nDec(1)) ',' num2str(nDec(2)) ])
sw.add(')$; ')
sw.add('the other construction parameters are summarized in ')
sw.addcr(sprintf('Table~\\ref{tab:%02d}.}',ntabref))
sw.addcr(sprintf('\\label{fig:%02d}',nfig))
sw.addcr('\end{figure}')
sw.add('\endinput')

%%
disp(sw)
write(sw,sprintf('materials/fig%02d.tex',nfig));
