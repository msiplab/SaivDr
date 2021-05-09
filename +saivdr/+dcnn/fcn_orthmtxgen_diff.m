function [matrix,matrixpst,matrixpre] = fcn_orthmtxgen_diff(...
    angles,mus,pdAng,matrixpst,matrixpre)
%FCN_ORTHMTXGEN_DIFF
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
if pdAng < 1
    if isempty(angles)
        matrixpst = eye(nDim_);
        matrixpre = eye(nDim_);
    else
        matrixpst = eye(nDim_,'like',angles);
        matrixpre = eye(nDim_,'like',angles);
    end
    if isdlarray(angles)
        matrixpst = dlarray(matrixpst);
        matrixpre = dlarray(matrixpre);
    end
else
    if isempty(angles)
        matrixrev = eye(nDim_);
        matrixdif = zeros(nDim_);
    else
        matrixrev = eye(nDim_,'like',angles);
        matrixdif = zeros(nDim_,'like',angles);
    end
    if isdlarray(angles)
        matrixrev = dlarray(matrixrev);
        matrixdif = dlarray(matrixdif);
    end
end

if ~isempty(angles)
    if pdAng == 0 % Initialization
        iAng = 1;
        vt = matrixpst(1,:);
        vb = matrixpst(2,:);
        angle = angles(iAng);
        if isdlarray(angles)
            [vt,vb] = dlrot_(vt,vb,angle);            
        else
            [vt,vb] = rot_(vt,vb,angle);
        end
        matrixpst(1,:) = vt;
        matrixpst(2,:) = vb;
        for iTop=1:nDim_-1
            vt = matrixpst(iTop,:);
            for iBtm=iTop+1:nDim_
                if iAng > 1
                    vb = matrixpst(iBtm,:);                    
                    angle = angles(iAng);
                    if isdlarray(angles)                    
                        [vt,vb] = dlrot_(vt,vb,angle);                                            
                    else
                        [vt,vb] = rot_(vt,vb,angle);                    
                    end
                    matrixpst(iBtm,:) = vb;
                end
                iAng = iAng + 1;
            end
            matrixpst(iTop,:) = vt;
        end
        matrix = matrixpst;
    else % Sequential differentiation
        iAng = 1;
        for iTop=1:nDim_-1
            rt = matrixrev(iTop,:);
            dt = zeros(1,nDim_);
            dt(iTop) = 1;
            for iBtm=iTop+1:nDim_
                if iAng == pdAng                
                    angle = angles(iAng);                    
                    % 
                    rb = matrixrev(iBtm,:);
                    if isdlarray(angles)                                        
                        [rt,rb] = dlrot_(rt,rb,-angle);                        
                    else
                        [rt,rb] = rot_(rt,rb,-angle);
                    end
                    matrixrev(iTop,:) = rt;
                    matrixrev(iBtm,:) = rb;
                    %
                    db = zeros(1,nDim_);
                    db(iBtm) = 1;                    
                    dangle = angle + pi/2;
                    if isdlarray(angles)
                        [dt,db] = dlrot_(dt,db,dangle);
                    else
                        [dt,db] = rot_(dt,db,dangle);
                    end
                    matrixdif(iTop,:) = dt;
                    matrixdif(iBtm,:) = db;
                    %
                    matrixpst = matrixpst*matrixrev;
                    matrix    = matrixpst*matrixdif*matrixpre;
                    matrixpre = matrixrev.'*matrixpre;
                end
                iAng = iAng + 1;
            end
        end
    end
end
if isscalar(mus)
    matrix = mus*matrix;
elseif ~isempty(mus)
    if isdlarray(angles)
        matrix = mus(:).* matrix;
    else
        matrix = bsxfun(@times,mus(:),matrix);
    end
end
end

function [vt,vb] = rot_(vt,vb,angle)
c = cos(angle); 
s = sin(angle); 
u  = bsxfun(@plus,vt,vb);
u  = bsxfun(@times,s,u);
vt = bsxfun(@times,c+s,vt);
vb = bsxfun(@times,c-s,vb);
vt = bsxfun(@minus,vt,u);
vb = bsxfun(@plus,vb,u);
end

function [vt,vb] = dlrot_(vt,vb,angle)
c = cos(angle); 
s = sin(angle); 
u  = vt + vb;
u  = s * u;
vt = (c + s) * vt;
vb = (c - s) * vb;
vt = vt - u;
vb = vb + u;
end