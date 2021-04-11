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
% Copyright (c) 2020, Shogo MURAMATSU
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
if isempty(angles)
    matrix = zeros(nDim_);
else
    matrix = zeros(nDim_,'like',angles);
end
for idx = 1:nDim_
    matrix(idx,idx) = 1;
end
v = zeros(2,nDim_,'like',angles);
TOP = 1;
BTM = 2;
if ~isempty(angles)
    iAng = 1;
    for iTop=1:nDim_-1
        %vt = matrix(iTop,:);
        v(TOP,:) = matrix(iTop,:);
        for iBtm=iTop+1:nDim_
            angle = angles(iAng);
            if iAng == pdAng
                angle = angle + pi/2;
            end
            c = cos(angle); %
            s = sin(angle); %
            %vb = matrix(iBtm,:);
            v(BTM,:) = matrix(iBtm,:);
            %{
            u  = s*(vt + vb);
            vt = (c + s)*vt;
            vb = (c - s)*vb;
            vt = vt - u;
            %}
            v = [ c -s ; s c ] * v;
            if iAng == pdAng
                matrix = 0*matrix;
            end
            %matrix(iBtm,:) = vb + u;
            matrix(iBtm,:) = v(BTM,:);
            %
            iAng = iAng + 1;
        end
        %matrix(iTop,:) = vt;
        matrix(iTop,:) = v(TOP,:);
    end
end
if isscalar(mus)
    matrix = mus*matrix;
elseif ~isempty(mus)
    for idx = 1:nDim_
        matrix(idx,:) = mus(idx)*matrix(idx,:);
    end
end
end

