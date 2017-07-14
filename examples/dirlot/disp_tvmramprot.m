% DISP_TVMRAMPROT Display evaluation results of TVM for ramp rotation image
%
% This script displays the evaluation result of TVM characteristics 
% obtained by the script MAIN_TVMRAMPROT.
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014-2017, Shogo MURAMATSU
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

filename = 'results/tvmramprot_results';
if exist(filename,'file')
    load results/tvmramprot_results
else
    error('tvmramprot_results.mat is required. Please execute main_tvmrmprot.m first.')
end

%%
import saivdr.dictionary.nsgenlotx.NsGenLotUtility
phixd = 30;
dim = [ 32 32 ];
disp(phixd);
srcImg = NsGenLotUtility.trendSurface(phixd,dim);
figure(1), imshow(srcImg)
imwrite(srcImg,sprintf('images/ramp%03d.tif',phixd))

%% R_0
[mx,my] = meshgrid(-45:134,-45:134);
figure(2)
stepDeg = 1;
surfc(mx(1:stepDeg:end,1:stepDeg:end),...
    my(1:stepDeg:end,1:stepDeg:end),...
    arrayNorm0woScl(1:stepDeg:end,1:stepDeg:end)/arrayCard);
xlabel('\phi_x','FontSize',12,'FontName','AvantGrade')
ylabel('\phi','FontSize',12,'FontName','AvantGrade')
zlabel('w_0','FontSize',12,'FontName','AvantGrade')
axis([-45 135 -45 135 0 1])
colormap gray
caxis([ 0 1 ])
view(-20,20)
set(gca,'YTick',[-45 0 45 90 135]);
set(gca,'YTickLabel',{'-pi/4', '0', 'pi/4', 'pi/2', '3pi/4'});
set(gca,'XTick',[-45 0 45 90 135]);
set(gca,'XTickLabel',{'-pi/4', '0', 'pi/4', 'pi/2', '3pi/4'});
set(gca,'FontSize',12,'FontName','AvantGrade')
shading interp

%% For psfrag
%{
[mx,my] = meshgrid(-45:134,-45:134);
figure(3)
step = 2;
offset = 1;
mesh(mx(1+offset:step:end,1+offset:step:end),...
    my(1+offset:step:end,1+offset:step:end),...
    arrayNorm0woScl(1+offset:step:end,1+offset:step:end)/arrayCard);
xlabel('x','FontSize',12,'FontName','AvantGrade')
ylabel('y','FontSize',12,'FontName','AvantGrade')
zlabel('z','FontSize',12,'FontName','AvantGrade')
axis([-45 135 -45 135 0 1])
colormap gray
caxis([ 0 1 ])
view(-20,20)
set(gca,'YTick',[-45 0 45 90 135]);
set(gca,'YTickLabel',{'a', 'b', 'c', 'd', 'e'});
set(gca,'XTick',[-45 0 45 90 135]);
set(gca,'XTickLabel',{'a', 'b', 'c', 'd', 'e'})
%set(gca,'ZTick',[0 200 400 600]);
set(gca,'FontSize',12,'FontName','AvantGrade')
%shading interp
%}
