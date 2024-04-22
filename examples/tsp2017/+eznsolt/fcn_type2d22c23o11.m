function [h,f,E] = fcn_type2d22c23o11(angs,mus,isnodc)
%
% $Id: fcn_type2d22c23o11.m 850 2015-10-29 21:19:55Z sho $
% 
nChs = 5;

if nargin < 3
    isnodc = false;
end

if nargin < 2
    mus  = [ 1 1  1 1  1 -1  -1 1 1 ].'; 
end

if nargin < 1
    angs = [ 0    0   -pi/2  -pi/2 0 0 ].';
end

%% Preperation
import saivdr.dictionary.utility.PolyPhaseMatrix2d
import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem

omg4 = OrthonormalMatrixGenerationSystem();

%% Delay chain
clear d_M
d_M(1,1,1,1) = 1;
d_M(2,1,2,1) = 1;
d_M(3,1,1,2) = 1;
d_M(4,1,2,2) = 1;
d_M = PolyPhaseMatrix2d(d_M);

%% Constraction
nAngU = 3;
nMusU = 3;
aidx  = 4;
midx  = 7;

%% E0
if isnodc
    angs(1) = 0;
end
angs_(1:3) = angs(1:3);
angs_(4)   = 0;
mus_(1:6)  = mus(1:6);
mus_(7)     = 1;
mus_(8)     = 1;
[~,~,E0] = eznsolt.fcn_type1d22c22([1 1],angs_,mus_,false);
U0 = step(omg4,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
%
R0 = blkdiag(eye(2),U0(:,1:2));
E  = R0*E0;

%% 
Eu = upsample(E,[2 2],[1 2]);
H = Eu*d_M;

%% Extraction of filters
h = zeros(size(double(H),3),size(double(H),4),nChs);
for aidx = 1:nChs
    h(:,:,aidx) = H(aidx,1,:,:);
end
f = flip(flip(h,1),2);