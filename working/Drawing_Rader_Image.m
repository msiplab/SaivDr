
% ヲ

%% --------row--------------------------------------

%load('raderimage.mat')
img = im/max(abs(im(:)));
r = cropR;
x = cropX;

%% -----�?----------------------------------------

% ヲ
figure
clims = [-60 0];
imagesc(r, x, 20*log10(abs(img)), clims)
xlabel('Range')
ylabel('Runing Distance')
xlim([0 8])
colorbar

figure
clims = [-pi pi];
imagesc(r, x, angle(img), clims)
xlabel('Range')
ylabel('Runing Distance')
xlim([0 8])
colorbar

