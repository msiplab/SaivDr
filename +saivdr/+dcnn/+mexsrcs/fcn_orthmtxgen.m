function matrix = fcn_orthmtxgen(angles,mus,useGpu,isLessThanR2021b) %#codegen
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
if nargin < 4
    isLessThanR2021b = false;
end
if nargin < 3
    useGpu = isgpuarray(angles);
end
nDim_ = (1+sqrt(1+8*length(angles)))/2;
matrix = eye(nDim_,'like',angles);
if ~isempty(angles)
    iAng = uint32(1);
    for iTop=1:nDim_-1
        vt = matrix(iTop,:);
        for iBtm=iTop+1:nDim_
            angle = angles(iAng);
            if angle ~= 0
                c = cos(angle);
                s = sin(angle);
                vb = matrix(iBtm,:);
                if useGpu           
                    u  = arrayfun(@(s,vt,vb) s.*(vt+vb),s,vt,vb);
                    vt = arrayfun(@(c,s,vt,u) (c+s).*vt-u,c,s,vt,u);
                    matrix(iBtm,:) = arrayfun(@(c,s,vb,u) (c-s).*vb+u,c,s,vb,u);
                elseif isLessThanR2021b % on CPU
                    u  = bsxfun(@times,s,bsxfun(@plus,vt,vb));
                    vt = bsxfun(@minus,bsxfun(@times,c+s,vt),u);
                    matrix(iBtm,:) = bsxfun(@plus,bsxfun(@times,c-s,vb),u);
                else % on CPU                 
                    u  = s.*(vt+vb);
                    vt = (c+s).*vt-u;
                    matrix(iBtm,:) = (c-s).*vb+u;
                end
            end
            %
            iAng = iAng + 1;
        end
        matrix(iTop,:) = vt;
    end
end
if ~all(mus==1) 
    if useGpu
        matrix = arrayfun(@times,mus(:),matrix);
    elseif isLessThanR2021b % on CPU
        matrix = bsxfun(@times,mus(:),matrix);
    else % on CPU
        matrix = mus(:).*matrix;
    end
end
end
