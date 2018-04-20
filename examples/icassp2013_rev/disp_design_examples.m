% DISP_DESIGN_EXAMPLES Display design examples
%
% This script displays some design examples of nsoltx.
% The first example was designed by using the script MIAN_PARNSOLTDSGN.
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014-2016, Shogo MURAMATSU
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
clear all; close all

%% Initial setting
example = 1; % Select a value from {1,2,3}
%
if example == 1      % Design example of Type-II NSOLT
    nDec = [ 2 2 ];
    nChs = [ 5 2 ];
    nOrd = [ 4 4 ];
elseif example == 2  % Critically-sampled Haar transform
    nDec = [ 2 2 ];
    nChs = [ 2 2 ];
    nOrd = [ 0 0 ];
else                 % Non-subsampled Haar transform
    nDec = [ 1 1 ];
    nChs = [ 2 2 ];
    nOrd = [ 1 1 ];
end
nVm  = 1; % # of vanishing moments
location = './filters'; % Location of MAT file
import saivdr.dictionary.utility.Direction
sparam = sprintf('d%dx%d_c%d+%d_o%d+%d_v%d',...
    nDec(Direction.VERTICAL),nDec(Direction.HORIZONTAL),...
    nChs(Direction.VERTICAL),nChs(Direction.HORIZONTAL),...
    nOrd(Direction.VERTICAL),nOrd(Direction.HORIZONTAL),...
    nVm);

%% Load a design example
filterName = sprintf('%s/nsolt_%s.mat', location, sparam);
display(filterName)
S = load(filterName,'lppufb');
lppufb = saivdr.dictionary.utility.fcn_upgrade(S.lppufb);
prmMtcs = get(lppufb,'ParameterMatrixSet');
if isa(lppufb,'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeIISystem')
    fprintf('W_0 = \n');
    disp(step(prmMtcs,[],uint32(1)))
    fprintf('U_0 = \n');
    disp(step(prmMtcs,[],uint32(2)))
    for idx = uint32(3):2:get(prmMtcs,'NumberOfParameterMatrices')
        if (idx+1)/2 < nOrd(Direction.HORIZONTAL)
            direction = 'x';
            offset = 0;
        else
            direction = 'y';
            offset = nOrd(Direction.HORIZONTAL)/2;
        end
        fprintf('W^{%s}_%d = \n',direction,(idx-1)/2-offset);
        disp(step(prmMtcs,[],uint32(idx)))
        fprintf('U^{%s}_%d = \n',direction,(idx-1)/2-offset);
        disp(step(prmMtcs,[],uint32(idx+1)))
    end
else
    fprintf('W_0 = \n')
    disp(step(prmMtcs,[],uint32(1)))
    fprintf('U_0 = \n')
    disp(step(prmMtcs,[],uint32(2)))
    for idx = uint32(3):get(prmMtcs,'NumberOfParameterMatrices')
        if idx-2 <= nOrd(Direction.HORIZONTAL)
            direction = 'x';
            offset = 0;
        else
            direction = 'y';
            offset = nOrd(Direction.HORIZONTAL);
        end        
        fprintf('U^{%s}_%d = \n',direction,idx-2-offset)
        disp(step(prmMtcs,[],idx))
    end
end

%% Frequency response
release(lppufb)
set(lppufb,'OutputMode','SynthesisFilters');
F = step(lppufb,[],[]);
h = figure;
pos = get(h,'Position');
prewidth = pos(3);
height = pos(4);
pstwidth = max(nChs)*height/2;
prex = pos(1);
pstx = prex - (pstwidth-prewidth)/2;
pos(1) = pstx;
pos(3) = pstwidth;
set(h,'Position',pos);
for idx=1:sum(nChs)
    subplot(2,max(nChs),idx);
    freqz2(F(:,:,idx))
    view(-37.5,60)
    colormap('default')
end
%{
for idx = 1:sum(nChs)
    h = figure(idx);
    freqz2(F(:,:,idx))
    xlabel('x','FontSize',80,'FontName','AvantGrade')
    ylabel('y','FontSize',80,'FontName','AvantGrade')
    %zlabel('Magnitude','FontSize',24,'FontName','AvantGrade')
    zlabel('','FontSize',80,'FontName','AvantGrade')
    set(gca,'XTick',[1]);
    set(gca,'YTick',[-1 1]);
    set(gca,'ZTick',[]);
    set(gca,'FontSize',80,'FontName','AvantGrade')
    grid off
    % axis off
    view(-37.5,60)
    %colormap(gray)
    print(h,'-depsc2',['./results/frq' sparam '_' num2str(idx) '.eps'])
end
%}

%% Atomic images
figure
atmimshow(lppufb)
%{
for iSub = 1:sum(nChs)
    imwrite(imresize(rot90(F(:,:,iSub),2)+0.5,4,'nearest'),...
        ['./results/atmim' sparam '_' num2str(iSub) '.tif']);
end
%}
