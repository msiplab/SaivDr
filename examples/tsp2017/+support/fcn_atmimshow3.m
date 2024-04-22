function fcn_atmimshow3( f, fname )
%FCN_ATMIMSHOW3 Display impluse responses of filters
%
% SVN identifier:
% $Id: fcn_atmimshow3.m 850 2015-10-29 21:19:55Z sho $
%
% Requirements: MATLAB R2013b
%
% Copyright (c) 2014-2015, Kosuke FURUYA and Shogo MURAMATSU
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
isExport = true;
%
H = padarray(f,[2 2 2]);
%
cmn = min(H(:));
cmx = max(H(:));
ly = size(H,1);
lx = size(H,2);
lz = size(H,3);
%
[x,y,z] = meshgrid(-(lx-1)/2:(lx-1)/2,-(ly-1)/2:(ly-1)/2,-(lz-1)/2:(lz-1)/2);
%
nChs = size(f,4);
for ib=1:sum(nChs)
    hold off
    %
    v = H(:,:,:,ib);
    v = 2*(v-cmn)/(cmx-cmn)-1;
    xslice = -(lx-1)/2:.125:(lx-1)/2;
    yslice = -(ly-1)/2:.125:(ly-1)/2;
    zslice = -(lz-1)/2:.125:(lz-1)/2;
    figure(ib)
    hslice = slice(x,y,z,v,xslice,yslice,zslice,'nearest');
    set(hslice,'FaceColor','interp',...
        'EdgeColor','none',...
        'DiffuseStrength',.8);
    caxis([-1 1]);
    axis equal
    axis vis3d
    colormap(gray)
    for iSlice = 1:length(hslice)
        map = abs(get(hslice(iSlice),'CData'));
        set(hslice(iSlice),...
           'AlphaDataMapping','scaled',...
           'AlphaData',map.^2,... 1.6,...
           'FaceAlpha','texture',...
           'FaceColor','texture');
    end
    set(gca,'XTickLabel',[])
    set(gca,'YTickLabel',[])
    set(gca,'ZTickLabel',[])
    set(gca,'TickLength',[0 0])
    %
    set(gca,'LineWidth',2)
    set(gcf,'Color','white')
    if isExport
        h = getframe(gcf);
        im = rgb2gray(frame2im(h));
        im = im(31:380,121:110+350);
        id = ('a'-1')+ib;
        imwrite(im,sprintf('%s%c.tif',fname,id))
    end
end
end