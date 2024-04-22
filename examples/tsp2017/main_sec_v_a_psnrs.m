%MAIN_SEC_V_A_PSNRS Sparse approximation results with 2-D NSOLTs (Table IV)
%
% This script was used for summarising the sparse approximation
% performances with 2-D NSOLTS in Section V.A.
%
% The following material is also reproduced:
%
%  * materials/tab04.tex
%
% SVN identifier:
% $Id: main_sec_v_a_psnrs.m 856 2015-11-02 02:01:16Z sho $
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
ntab     = 4; % Table number
nfigref1 = 6; % Figure reference number of training image
nfigref2 = 8; % Figure reference number of atomic images

setpath

%% Parameters
nDec = [ 2 2 ]; % Downsampling factor [M0 M1]
maxOrd = 3;
nChsSp = [ 3 3 ]; % # of channels [ ps pa ](Separable OLT)
nChsNs = [        % # of channels [ ps pa ](NSOLT)
    2 2 ;
    2 3 ;
    3 2 ;
    3 3 ];
idx = 1;
% Separable OLT
clear params
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
strImg   = 'barbara64'; % Training image
sparsity = 1/4;         % Sparsity in ratio

%%
sw = StringWriter();

sw.addcr('%#! latex double')
sw.addcr('%')
sw.addcr('% $Id: main_sec_v_a_psnrs.m 856 2015-11-02 02:01:16Z sho $')
sw.addcr('%')
sw.addcr('\begin{table}[tb]')
sw.addcr('\centering')
sw.addcr('\caption{PSNRs [dB] of sparse approximation results through')
sw.add('IHT with the SPOLTs and NSOLTs shown in ')
sw.add(sprintf('Fig.~\\ref{fig:%02d} ',nfigref2))
sw.add(' for the training image shown in ')
sw.add(sprintf('Fig.~\\ref{fig:%02d}, where ``SP'''' means ``separable.''''}',nfigref1))
sw.addcr(sprintf('\\label{tab:%02d}',ntab))
sw.addcr('\begin{tabular}{|c|c|c|c||c|c|c|}\hline')
sw.addcr('\multirow{2}{*}{Type} & \multicolumn{2}{c|}{$P$} & Redundancy & \multicolumn{3}{c|}{$\bar{\mathbf{n}}=(N_0,N_1)^T$} \\ \cline{2-3}\cline{5-7}')
sw.addcr('& $p_\textrm{s}$ & $p_\textrm{a}$ & $R=P/M$ & $(1,1)$ & $(2,2)$ & $(3,3)$ \\ \hline\hline')
%op
pps = 0;
ppa = 0;
for idx = 1:length(params)
    nDec = params{idx}.nDec;
    nOrd = params{idx}.nOrd;
    ps = params{idx}.nChs(1);
    pa = params{idx}.nChs(2);
    opt = params{idx}.opt;
    if nOrd(1) == 1 && nOrd(2) == 1
        if strcmp(opt,'sep')
            sw.add(' SP &')
        elseif ps == pa
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
    s = load(sprintf('results/eznsolt_d%d%dc%d%do%d%d%s_ndc1_%s',...
        nDec(1),nDec(2),ps,pa,nOrd(1),nOrd(2),opt,strImg),'resPsnr','h');
    psnr = s.resPsnr;
    %f    = flip(flip(s.h,1),2);
    id = ('a'-1)+idx;
    sw.add(sprintf('& (%c) %5.2f ',id,psnr))
    %
    if nOrd(1) == maxOrd && nOrd(2) == maxOrd
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