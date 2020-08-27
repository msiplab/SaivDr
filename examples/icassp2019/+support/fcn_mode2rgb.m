function [watermode,bedmode] = fcn_mode2rgb(modearray,scale)
%FCN_MODE2RGB Convert modearray to color map
%  

% Water mode
array = modearray(:,:,1);
h = angle(array);
s = ones(size(array));
v = scale*abs(array);
watermode = hsv2rgb(cat(3,h,s,v));

% Bed mode
array = modearray(:,:,2);
h = angle(array);
s = ones(size(array));
v = scale*abs(array);
bedmode = hsv2rgb(cat(3,h,s,v));

end

