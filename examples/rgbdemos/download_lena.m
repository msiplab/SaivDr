fname = 'lena.png';

if ~exist(sprintf('./%s',fname),'file')
    img = imread(...
        sprintf('http://homepages.cae.wisc.edu/~ece533/images/%s',...
        fname));
    imwrite(img,sprintf('./%s',fname));
    fprintf('Downloaded and saved %s.\n',fname);
else
    fprintf('%s already exists.\n',fname);
end
