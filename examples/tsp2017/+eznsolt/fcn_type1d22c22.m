function [h,f,E] = fcn_type1d22c22(ord,angs,mus,isnodc)
%
% $Id: fcn_type1d22c22.m 850 2015-10-29 21:19:55Z sho $
% 
if nargin < 4
    isnodc = false;
end

if nargin < 3
    mus  = [ 1 1  1 1  1 -1  -1 1 ].'; 
end

if nargin < 2
    angs = [ 0  0  -pi/2 -pi/2 ].';
end

if nargin < 1
    ord = [1 1];
end

%% Preperation
import saivdr.dictionary.utility.PolyPhaseMatrix2d
import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem

omg2 = OrthonormalMatrixGenerationSystem();

%% Delay chain
clear d_M
d_M(1,1,1,1) = 1;
d_M(2,1,2,1) = 1;
d_M(3,1,1,2) = 1;
d_M(4,1,2,2) = 1;
d_M = PolyPhaseMatrix2d(d_M);

%% 2-D DCT
C_MJ_M = [ 1  1  1  1 ;
        1 -1 -1  1 ;
       -1  1 -1  1 ;
       -1 -1  1  1 ]/2;

%% Butterfly

B = [ eye(2)  eye(2) ; eye(2) -eye(2) ]/sqrt(2);
  
%% Lambda
clear Ly;
Ly(:,:,1,1) = [ eye(2) zeros(2) ; zeros(2) zeros(2) ]; 
Ly(:,:,2,1) = [ zeros(2) zeros(2) ; zeros(2) eye(2) ]; 
Ly = PolyPhaseMatrix2d(Ly);

clear Lx;
Lx(:,:,1,1) = [ eye(2) zeros(2) ; zeros(2) zeros(2) ]; 
Lx(:,:,1,2) = [ zeros(2) zeros(2) ; zeros(2) eye(2) ]; 
Lx = PolyPhaseMatrix2d(Lx);

%% Qd
Qy = B*Ly*B;
Qx = B*Lx*B;

%% Constraction
nAngW = 1;
nAngU = 1;
nMusW = 2;
nMusU = 2;
aidx=1;
midx=1;

%% E0
if isnodc
    angs(1) = 0;
    mus(1)  = 1;
end
W0  = step(omg2,angs(aidx:aidx+nAngW-1),mus(midx:midx+nMusW-1));
aidx = aidx+nAngW;
midx = midx+nMusW;
%
U0  = step(omg2,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
aidx = aidx+nAngU;
midx = midx+nMusU;
%
R0 = blkdiag(W0(:,1:2),U0(:,1:2));
E = R0*C_MJ_M;

%% Vertical extension
for ordY = 1:ord(1)
    U = step(omg2,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
    aidx = aidx+nAngU;
    midx = midx+nMusU;
    R = blkdiag(eye(2),U);
    E  = R*Qy*E;
end

%% Horizontal extension
for ordX = 1:ord(2)
    U = step(omg2,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
    aidx = aidx+nAngU;
    midx = midx+nMusU;
    R = blkdiag(eye(2),U);
    E  = R*Qx*E;
end

%% 
if ord(1) == 0 && ord(2) == 0
    Eu = E;
else
    Eu = upsample(E,[2 2],[1 2]);
end
H = Eu*d_M;

%% Extraction of filters
h = zeros(size(double(H),3),size(double(H),4),4);
for idx = 1:4
    h(:,:,idx) = H(idx,1,:,:);
end
f = flip(flip(h,1),2);