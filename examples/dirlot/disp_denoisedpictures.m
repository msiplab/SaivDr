close all
%%
sigma = 30;
imgset = { 'goldhill' 'lena' 'barbara' 'baboon' };
trxset = { 'sym5' 'son4' 'udn4' };
nImgs = length(imgset);
nTrxs = length(trxset);
ppart{1} = [196 196];
ppart{2} = [196 196];
ppart{3} = [196 196];
ppart{4} = [16 128];
psize = [128 128];
for iImg = 1:nImgs
    f = figure;
    set(f,'Name', imgset{iImg})
    % Original
    fname = sprintf('./images/%s.png',imgset{iImg});
    pic = imread(fname);
    pic = circshift(pic,-ppart{iImg});
    %
    pic = pic(1:psize(1),1:psize(2));
    imwrite(pic,sprintf('./images/fig_%s.tif',imgset{iImg}));
    subplot(2,3,1)
    subimage(pic)
    title('Original')
    axis off
    % Noisy image
    fname = sprintf('./images/noisy_%s_%d_0.tif',imgset{iImg},sigma);
    pic = imread(fname);
    pSize = size(pic);
    pic = circshift(pic,-ppart{iImg});
    %
    pic = pic(1:psize(1),1:psize(2));
    imwrite(pic,sprintf('./images/fig_noisy_%s_%d.tif',imgset{iImg},sigma));
    subplot(2,3,2)
    subimage(pic)
    title('Noisy image')
    axis off
    for iTrx = 1:nTrxs
        %
        fname = sprintf('./images/result_%s_%s_%d.tif',...
            trxset{iTrx},imgset{iImg},sigma);
        pic = imread(fname);
        pSize = size(pic);
        pic = circshift(pic,-ppart{iImg});
        %
        pic = pic(1:psize(1),1:psize(2));
        imwrite(pic,sprintf('./images/fig_%s_%s_%d.tif',...
            trxset{iTrx},imgset{iImg},sigma));
        subplot(2,3,iTrx+3)
        subimage(pic)
        title(trxset{iTrx})
        axis off
    end
end
