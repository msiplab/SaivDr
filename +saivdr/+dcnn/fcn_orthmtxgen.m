function matrix = fcn_orthmtxgen(angles,mus,pdAng)
%FCN_ORTHMTXGEN
%
% Function realization of
% saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
% for supporting dlarray (Deep learning array for custom training
% loops)
%
% Requirements: MATLAB R2020a
%
% Copyright (c) 2020-2021, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/

if nargin < 3
    pdAng = 0;
end

nDim_ = (1+sqrt(1+8*length(angles)))/2;
if isdlarray(angles)
    matrix = dlarray(eye(nDim_,angles.underlyingType));
else
    matrix = eye(nDim_,'like',angles);
end
if any(angles) || ~isempty(angles)
    iAng = 1;
    for iTop=1:nDim_-1
        vt = matrix(iTop,:);
        for iBtm=iTop+1:nDim_
            angle = angles(iAng);
            if iAng == pdAng
                angle = angle + pi/2;
            end
            if angle ~= 0
                c = cos(angle);
                s = sin(angle);
                vb = matrix(iBtm,:);
                if isdlarray(angles)
                    u = s * ( vt + vb );
                else
                    u  = bsxfun(@times,s,bsxfun(@plus,vt,vb));
                end
                if iAng == pdAng
                    matrix = zeros(size(matrix),'like',matrix);
                end
                if isdlarray(angles)
                    vt = ( c + s ) * vt - u;
                    matrix(iBtm,:) = ( c - s ) * vb + u;
                else
                    vt = bsxfun(@minus,bsxfun(@times,c+s,vt),u);
                    matrix(iBtm,:) = bsxfun(@plus,bsxfun(@times,c-s,vb),u);
                end
            end
            %
            iAng = iAng + 1;
        end
        matrix(iTop,:) = vt;
    end
end
if isscalar(mus)
    matrix = mus * matrix;
elseif ~all(mus==1) || ~isempty(mus)
    if isdlarray(angles)
        matrix = mus(:) .* matrix;
    else
        matrix = bsxfun(@times,mus(:),matrix);
    end
end
end
