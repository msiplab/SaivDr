%MAIN_SEC_V_B_PSNRS Sparse approximation results with 3-D NSOLTs (Table V)
%
% This script was used for summarising the sparse approximation
% performances with 3-D NSOLTs in Section V.B.
%
% The following material is also reproduced:
%
%  * materials/tab05.tex
%
% SVN identifier:
% $Id: main_sec_v_b_psnrs.m 856 2015-11-02 02:01:16Z sho $
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
% Copyright (c) 2014, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
ntab     = 5;  % Table number
nfigref1 = 9;  % Figure reference number of training volume data

setpath

%% Parameters
nDec = [ 2 2 2 ]; % Downsampling factor [M0 M1]
nOrd = [ 2 2 2 ];
nChsNs = [        % # of channels [ ps pa ](NSOLT)
    4 4 ;
    4 5 ;
    5 4 ;
    5 5 ];
% NSOLT
idx = 1;
clear params
for iChs = 1:size(nChsNs,1)
    params{idx} = struct(...
        'nDec',nDec,...
        'nChs',nChsNs(iChs,:),...
        'nOrd',nOrd);
    idx = idx+1;
end
strImg   = 'mrbrain32x32x32'; % Training volume data
sparsity = 1/4;         % Sparsity in ratio

%%
sw = StringWriter();

sw.addcr('%#! latex double')
sw.addcr('%')
sw.addcr('% $Id: main_sec_v_b_psnrs.m 856 2015-11-02 02:01:16Z sho $')
sw.addcr('%')
sw.addcr('\begin{table}[tb]')
sw.addcr('\centering')
sw.addcr('\caption{PSNRs [dB] of sparse approximation results through')
sw.add('IHT with the NSOLTs designed ')
sw.add(' for the training volume data shown in ')
sw.addcr(sprintf('Fig.~\\ref{fig:%02d},',nfigref1))
sw.add('where $\mathbf{M}=\mathrm{diag}(M_0,M_1,M_2)=\mathrm{diag}(')
sw.add([ num2str(nDec(1)) ',' num2str(nDec(2)) ',' num2str(nDec(3)) ])
sw.add(')$ and ')
sw.add('$\bar{\mathbf{n}}=(N_0,N_1,N_2)^T=(')
sw.add([ num2str(nOrd(1)) ',' num2str(nOrd(2)) ',' num2str(nOrd(3)) ])
sw.addcr(')^T$.}')
sw.addcr(sprintf('\\label{tab:%02d}',ntab))
sw.addcr('\begin{tabular}{|c|c|c|c||c|}\hline')
sw.addcr('\multirow{2}{*}{Type} & \multicolumn{2}{c|}{$P$} & Redundancy & PSNR \\ \cline{2-3}')
sw.addcr('& $p_\textrm{s}$ & $p_\textrm{a}$ & $R=P/M$ & [dB]  \\ \hline\hline')
%
pps = 0;
ppa = 0;
for idx = 1:length(params)
    nDec = params{idx}.nDec;
    nOrd = params{idx}.nOrd;
    ps = params{idx}.nChs(1);
    pa = params{idx}.nChs(2);
    if nOrd(1) == 2 && nOrd(2) == 2 && nOrd(3) == 2
        if ps == pa
            sw.add('  I &')
        else
            sw.add(' II &')
        end
        P = ps+pa;
        sw.add(sprintf(' $%d$ & $%d$ & ',ps,pa))
        M = prod(nDec);
        if P == M
            sw.add(sprintf('   $%d$ ',1))
        else
            sw.add(sprintf(' $%d/%d$ ',P/gcd(P,M),M/gcd(P,M)))
        end
    end
    %
    s = load(sprintf('results/eznsolt_d%d%d%dc%d%do%d%d%d_ndc1_%s',...
        nDec(1),nDec(2),nDec(3),ps,pa,nOrd(1),nOrd(2),nOrd(3),strImg),'resPsnr','h');
    psnr = s.resPsnr;
    sw.add(sprintf('& %5.2f ',psnr))
    %
    if nOrd(1) == maxOrd && nOrd(2) == maxOrd && nOrd(3) == maxOrd
        sw.addcr('\\ \hline');
    end
    pps = ps;
    ppa = pa;
end
sw.addcr('\end{tabular}')
sw.addcr('\end{table}')
sw.add('\endinput')

%%
disp(sw)
write(sw,sprintf('materials/tab%02d.tex',ntab));