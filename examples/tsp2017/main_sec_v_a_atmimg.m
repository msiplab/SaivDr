%MAIN_SEC_V_A_ATMIMG Impluse responses of 2-D NSOLT synthesizer (fig. 8)
%
% This script was used for observing impluse responses of 2-D NSOLT 
% synthesizer shown in Section V.A.
% 
% The following materials are also reproduced:
%
% - tiff/fig08x.tif (x in {a,b,c,...,l})
% - materials/fig08.tex
%
% SVN identifier:
% $Id: main_sec_v_a_atmimg.m 849 2015-10-29 21:16:13Z sho $
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

nfig    = 8;    % Figure number
nfigref = 6;    % Figure reference number of training image
ntabref = 4'; % Table reference number of PSNR table

setpath

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

%% Sparsity
srcImg       = support.fcn_load_testimg2(strImg);
nSparseCoefs = round(numel(srcImg)*sparsity);

%% Observe atomic images
sw = StringWriter();

sw.addcr('%#! latex double')
sw.addcr('%')
sw.addcr('% $Id: main_sec_v_a_atmimg.m 849 2015-10-29 21:16:13Z sho $')
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
    s = load(sprintf('results/eznsolt_d%d%dc%d%do%d%d%s_ndc1_%s',...
        nDec(1),nDec(2),ps,pa,nOrd(1),nOrd(2),opt,strImg),'h');
    h = s.h;
    f = flip(flip(h,1),2);
    ids = idx;
    id = ('a'-1)+ids;
    dics = [];
    dica = [];
    for sidx = 1:(ps+pa)
        atm = padarray(f(:,:,sidx)+0.5,3-[nOrd(1) nOrd(2)],1);
        atm = padarray(atm,[1 1],1,'post');
        if sidx <= ps
            dics = [dics atm];
        else
            dica = [dica atm];
        end
    end
    if ps < pa
        dics  = padarray(dics,[0 4],1);
        dics  = padarray(dics,[0 1],1,'post');
    elseif pa < ps
        dica  = padarray(dica,[0 4],1);
        dica  = padarray(dica,[0 1],1,'post');
    end
    atmImg = padarray([dics ; dica],[1 1],1,'pre');
    figure(1),subplot(ceil(length(params)/3),3,ids),imshow(atmImg)
    fname = sprintf('fig%02d%c',nfig,id);
    imwrite(imresize(atmImg,4*size(atmImg),'nearest'),['tiff/' fname '.tif'])
    %
    sw.addcr('\begin{minipage}[b]{.237\linewidth}')
    sw.addcr('  \centering')
    sw.add('  \centerline{\includegraphics[height=16mm]{')
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
%
sw.add('\caption{Impluse responses of syntehsis filters designed through the SPOLT ')
sw.add('and NSOLT frameworks for the training image shown in ')
sw.add(sprintf('Fig.~\\ref{fig:%02d}, ',nfigref))
sw.add('where $\mathbf{M}=\mathrm{diag}(M_0,M_1)=\mathrm{diag}(')
sw.add([ num2str(nDec(1)) ',' num2str(nDec(2)) ])
sw.add(')$ and ')
sw.add('the other construction parameters are summarized in ')
sw.addcr(sprintf('Table~\\ref{tab:%02d}.}',ntabref))
sw.addcr(sprintf('\\label{fig:%02d}',nfig))
sw.addcr('\end{figure}')
sw.add('\endinput')

%%
disp(sw)
write(sw,sprintf('materials/fig%02d.tex',nfig));
