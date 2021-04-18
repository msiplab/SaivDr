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
        matrixpst = zeros(nDim_);
        matrixpre = zeros(nDim_);
    else
        matrixpst = zeros(nDim_,'like',angles);
        matrixpre = zeros(nDim_,'like',angles);
    end
    for idx = 1:nDim_
        matrixpst(idx,idx) = 1;
        matrixpre(idx,idx) = 1;
    end
else
    if isempty(angles)
        matrixrev = zeros(nDim_);
        matrixdif = zeros(nDim_);
    else
        matrixrev = zeros(nDim_,'like',angles);
        matrixdif = zeros(nDim_,'like',angles);
    end
    for idx = 1:nDim_
        matrixrev(idx,idx) = 1;
    end
end
if ~isempty(angles)
    if pdAng == 0 % Initialization
        iAng = 1;
        vt = matrixpst(1,:);
        vb = matrixpst(2,:);
        angle = angles(iAng);
        [vt,vb] = rot_(vt,vb,angle);
        matrixpst(1,:) = vt;
        matrixpst(2,:) = vb;
        for iTop=1:nDim_-1
            vt = matrixpst(iTop,:);
            for iBtm=iTop+1:nDim_
                if iAng > 1
                    vb = matrixpst(iBtm,:);                    
                    angle = angles(iAng);
                    [vt,vb] = rot_(vt,vb,angle);                    
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
                    [rt,rb] = rot_(rt,rb,-angle);
                    matrixrev(iTop,:) = rt;
                    matrixrev(iBtm,:) = rb;
                    %
                    db = zeros(1,nDim_);
                    db(iBtm) = 1;                    
                    dangle = angle + pi/2;
                    [dt,db] = rot_(dt,db,dangle);
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
    matrix = bsxfun(@times,mus(:),matrix);
end
end

function [vt,vb] = rot_(vt,vb,angle)
c = cos(angle); %
s = sin(angle); %
u  = bsxfun(@plus,vt,vb);
u  = bsxfun(@times,s,u);
vt = bsxfun(@times,c+s,vt);
vb = bsxfun(@times,c-s,vb);
vt = bsxfun(@minus,vt,u);
vb = bsxfun(@plus,vb,u);
end