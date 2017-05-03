% DISP_DESIGN_EXAMPLES Display design examples
%
% This script displays some design examples of DirLOT.
% The example was designed by using the script MIAN_PARDIRLOTDSGN.
%
% SVN identifier:
% $Id: disp_design_examples.m 748 2015-09-02 07:47:32Z sho $
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
close all

%% Initial setting
phi  = -30;  %{ [], -30, 0, 30, 60, 90, 120 };
nDec = [ 2 2 ];
nOrd = [ 4 4 ];
if isempty(phi)
    nVm  = 2; % # of vanishing moments
    fbname = 'nsgenlot';
    sVm  = 'v2';
else
    nVm  = 2; % # of vanishing moments
    fbname = 'dirlot';
    sVm  = sprintf('tvm%06.2f',phi);
end
location = './filters'; % Location of MAT file
import saivdr.dictionary.utility.Direction
sparam = sprintf('%s_d%dx%d_o%d+%d_%s',...
    fbname,...
    nDec(Direction.VERTICAL),nDec(Direction.HORIZONTAL),...
    nOrd(Direction.VERTICAL),nOrd(Direction.HORIZONTAL),...
    sVm);

%% Load a design example
filterName = sprintf('%s/%s.mat', location, sparam);
display(filterName)
S = load(filterName,'lppufb');
lppufb = S.lppufb;
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
pstwidth = prod(nDec)*height/2;
prex = pos(1);
pstx = prex - (pstwidth-prewidth)/2;
pos(1) = pstx;
pos(3) = pstwidth;
set(h,'Position',pos);
for idx=1:prod(nDec)
    subplot(2,max(nDec),idx);
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
