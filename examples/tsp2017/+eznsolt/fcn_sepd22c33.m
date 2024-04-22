function [h,f,E] = fcn_sepd22c33(ord,angs,mus,isnodc)
%
% $Id: fcn_sepd22c33.m 850 2015-10-29 21:19:55Z sho $
% 
if nargin < 4
    isnodc = false;
end

if nargin < 3
    mus  = [ 1 1 -1  1 1 1 1 -1 ].';
end

if nargin < 2
    angs = 0;
end

if nargin < 1
    ord = 1;
end

%% 1-D OLPPUFB Systems
angsh = angs;
musv = mus(1:2+ord);
mush = mus(3+ord:end);
%
if ord == 1
    hv = eznsolt.fcn_type1d2c11(1,musv,isnodc);
    hh = eznsolt.fcn_type2d2c12o1(angsh,mush,isnodc);
elseif ord == 2
    hv = eznsolt.fcn_type1d2c11(2,musv,isnodc);    
    hh = eznsolt.fcn_type2d2c12o2(angsh,mush,isnodc);
elseif ord == 3
    hv = eznsolt.fcn_type1d2c11(3,musv,isnodc);    
    hh = eznsolt.fcn_type2d2c12o3(angsh,mush,isnodc);
else
    error(['Not supported yet: ord = ' num2str(ord)])
end

%%
idx = 1;
h = zeros(2*(ord+1),2*(ord+1),6);
%
iCol = 1;
iRow = 1;
h(:,:,idx) = hv(:,iRow)*hh(:,iCol).';
idx = idx+1;
%
for iCol = 2:3
    iRow = 2;
    h(:,:,idx) = hv(:,iRow)*hh(:,iCol).';
    idx = idx+1;
end
%
iCol = 1;
iRow = 2;
h(:,:,idx) = hv(:,iRow)*hh(:,iCol).';
idx = idx+1;
%
for iCol = 2:3
    iRow = 1;
    h(:,:,idx) = hv(:,iRow)*hh(:,iCol).';
    idx = idx+1;
end
%
f = flip(flip(h,1),2);

%% 2-D PolyPhasMatrix
for idx = 1:6
    for iPhsH = 1:2
        for iPhsV = 1:2
            H(idx,2*(iPhsH-1)+iPhsV,:,:) = h(iPhsV:2:end,iPhsH:2:end,idx);
        end
    end
end
import saivdr.dictionary.utility.PolyPhaseMatrix2d
E = PolyPhaseMatrix2d(H);