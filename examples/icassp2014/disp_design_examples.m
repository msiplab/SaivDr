%DISP_DESIGN_EXAMPLES Display design examples
%
% This script displays some design examples of nsoltx.
% The example was designed by using the script MIAN_PARNSOLTDICLRN.
%
% SVN identifier:
% $Id: disp_design_examples.m 683 2015-05-29 08:22:13Z sho $
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
imgSet = { 'goldhill128', 'lena128', 'barbara128','baboon128' };
nChsSet  = { [ 4 4 ], [ 5 3 ], [ 6 2 ] };

%% Conditions
nDecs = [ 2 2 ];
nOrds = [ 4 4 ];
nCoefs = 2048;
%
imgName = imgSet{2};
nChs  = nChsSet{1};
nLevels = 3;
vm = 1;
%
isBest = false;
nIters = 15;

%%
fileName = sprintf(...
    'results/nsolt_d%dx%d_c%d+%d_o%d+%d_v%d_l%d_n%d_%s.mat',...
    nDecs(1),nDecs(2),nChs(1),nChs(2),...
    nOrds(1),nOrds(2),vm,nLevels,...
    nCoefs,imgName);
if exist(fileName,'file')
    display(fileName)
    S = load(fileName,'lppufbs','mses','bstidx','maxpsnr');
    lppufbs = S.lppufbs;
    if isBest
        bstlppufb = lppufbs{S.bstidx};
    elseif nIters < length(lppufbs)
        bstlppufb = lppufbs{nIters};
    else
        bstlppufb = lppufbs{end};
    end
    mses = S.mses;
    %
    figure(1)
    plot(sqrt(cell2mat(mses)))
    axis([ 1 length(mses) 0 10 ])
    xlabel('#Iter')
    ylabel('RMSE')
    grid on
    %
    hf2 = figure(2);
    pos = get(hf2,'Position');
    prewidth = pos(3);
    height = pos(4);
    pstwidth = max(nChs)*height/2;
    prex = pos(1);
    pstx = prex - (pstwidth-prewidth)/2;
    pos(1) = pstx;
    pos(3) = pstwidth;
    set(hf2,'Position',pos);
    clf
    bstlppufb = saivdr.dictionary.utility.fcn_upgrade(bstlppufb);
    release(bstlppufb)
    set(bstlppufb,'OutputMode','SynthesisFilters');
    F = step(bstlppufb,[],[]);
    for idx=1:sum(nChs)
        subplot(2,max(nChs),idx);
        freqz2(F(:,:,idx))
        view(-37.5,60)
        colormap('default')
    end
    %
    figure(3)
    clf
    atmimshow(bstlppufb)
    fprintf('MAX. PSNR = %6.2f [dB]\n',S.maxpsnr)
    for idx=1:sum(nChs)
        imwrite(F(:,:,idx)+0.5,sprintf('images/atom%02d.tif',idx))
    end    
else
    fprintf('%s : no such file\n',fileName)
end
