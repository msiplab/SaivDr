function n = kindndims(kind)
%KINDNDIMS Number of MATLAB dimensions of each kind of array
switch kind
    case {'image2d','coef2d','serial2d'}
        n = 4;
    case {'image3d','coef3d','serial3d'}
        n = 5;
    otherwise
        error('Unknown kind: %s',kind)
end
end
