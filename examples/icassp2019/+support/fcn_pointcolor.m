function ptcolor = fcn_pointcolor(z,basecolor,zrange)
%FCN_POINTCOLOR Colorization of points
% Colorization
frgb = basecolor/255;
fhsv = rgb2hsv(frgb);
zh = fhsv(1)*ones(size(z));
zs = fhsv(2)*ones(size(z));
zv = fhsv(3)+imfilter(z/zrange(2),...
    [8 0 0; 0 0 0 ; 0 0 -8],'sym');
ptcolor = im2uint8(hsv2rgb(cat(3,zh,zs,zv)));
end

