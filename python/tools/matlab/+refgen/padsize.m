function sz = padsize(x,ndims_)
%PADSIZE size(x) padded with trailing ones to ndims_ elements
sz = size(x);
sz = [sz ones(1,ndims_-numel(sz))];
end
