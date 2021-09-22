function matrix = fcn_orthmtxgen(angles,mus) %#codegen
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
                % TODO: Replace BSXFUN only for gpuArra                
                u  = bsxfun(@times,s,bsxfun(@plus,vt,vb));
                vt = bsxfun(@minus,bsxfun(@times,c+s,vt),u);
                matrix(iBtm,:) = bsxfun(@plus,bsxfun(@times,c-s,vb),u);
                % TODO: Use direct operations on CPU                    
                %u  = s.*(vt+vb);
                %vt = (c+s).*vt-u;
                %matrix(iBtm,:) = (c-s).*vb+u;
            end
            %
            iAng = iAng + 1;
        end
        matrix(iTop,:) = vt;
    end
end
if ~all(mus==1) 
    % TODO: Replace BSXFUN only for gpuArray     
    matrix = bsxfun(@times,mus(:),matrix);
    % TODO: Use direct operations on CPU       
    % matrix = mus(:).*matrix;
end
end
