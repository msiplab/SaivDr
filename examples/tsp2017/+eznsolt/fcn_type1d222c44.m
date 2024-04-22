function [h,f,E] = fcn_type1d222c44(ord,angs,mus,isnodc)
%
% $Id: fcn_type1d222c44.m 850 2015-10-29 21:19:55Z sho $
% 
if nargin < 4
    isnodc = false;
end

if nargin < 3
    mus  = [  ones(4,1) ; 
             -ones(4,1) ; 
             -ones(4,1) ; 
              ones(4,1) ;             
             -ones(4,1) ; 
              ones(4,1) ;             
             -ones(4,1) ; 
              ones(4,1) ];
end

if nargin < 2
    angs = zeros(6*8,1);
end

if nargin < 1
    ord = [ 2 2 2 ];
end

%% Preperation
import saivdr.dictionary.utility.PolyPhaseMatrix3d
import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem

omg4 = OrthonormalMatrixGenerationSystem();

%% Delay chain
clear d_M
d_M(1,1,1,1,1) = 1; % 000
d_M(2,1,2,1,1) = 1; % 100
d_M(3,1,1,2,1) = 1; % 000
d_M(4,1,2,2,1) = 1; % 000
d_M(5,1,1,1,2) = 1; % 000
d_M(6,1,2,1,2) = 1; % 000
d_M(7,1,1,2,2) = 1; % 000
d_M(8,1,2,2,2) = 1; % 111
d_M = PolyPhaseMatrix3d(d_M);

%% 3-D DCT
C_MJ_M = [           
            1           1           1           1           1           1           1           1 ;
           1           1          -1          -1          -1          -1           1           1 ;
           1          -1          -1           1           1          -1          -1           1 ;
           1          -1           1          -1          -1           1          -1           1 ;
          -1          -1          -1          -1           1           1           1           1 ;
          -1          -1           1           1          -1          -1           1           1 ;
          -1           1           1          -1           1          -1          -1           1 ;
          -1           1          -1           1          -1           1          -1           1 ]/sqrt(8);

%% Butterfly
p = 4;
B = [ eye(p)  eye(p) ; eye(p) -eye(p) ]/sqrt(2);
  
%% Lambda
clear Ly;
Ly(:,:,1,1,1) = [ eye(p) zeros(p) ; zeros(p) zeros(p) ]; 
Ly(:,:,2,1,1) = [ zeros(p) zeros(p) ; zeros(p) eye(p) ]; 
Ly = PolyPhaseMatrix3d(Ly);

clear Lx;
Lx(:,:,1,1,1) = [ eye(p) zeros(p) ; zeros(p) zeros(p) ]; 
Lx(:,:,1,2,1) = [ zeros(p) zeros(p) ; zeros(p) eye(p) ]; 
Lx = PolyPhaseMatrix3d(Lx);

clear Lz;
Lz(:,:,1,1,1) = [ eye(p) zeros(p) ; zeros(p) zeros(p) ]; 
Lz(:,:,1,1,2) = [ zeros(p) zeros(p) ; zeros(p) eye(p) ]; 
Lz = PolyPhaseMatrix3d(Lz);

%% Qd
Qy = B*Ly*B;
Qx = B*Lx*B;
Qz = B*Lz*B;

%% Constraction
nAngW = p*(p-1)/2;
nAngU = p*(p-1)/2;
nMusW = p;
nMusU = p;
aidx=1;
midx=1;

%% E0
if isnodc
    angs(1:p-1,1) = zeros(p-1,1);
    mus(1)  = 1;
end
W0  = step(omg4,angs(aidx:aidx+nAngW-1),mus(midx:midx+nMusW-1));
aidx = aidx+nAngW;
midx = midx+nMusW;
%
U0  = step(omg4,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
aidx = aidx+nAngU;
midx = midx+nMusU;
%
R0 = blkdiag(W0(:,1:4),U0(:,1:4));
E = R0*C_MJ_M;

%% Vertical extension
for ordY = 1:ord(1)
    U = step(omg4,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
    aidx = aidx+nAngU;
    midx = midx+nMusU;
    R = blkdiag(eye(p),U);
    E  = R*Qy*E;
end

%% Horizontal extension
for ordX = 1:ord(2)
    U = step(omg4,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
    aidx = aidx+nAngU;
    midx = midx+nMusU;
    R = blkdiag(eye(p),U);
    E  = R*Qx*E;
end

%% Depth extension
for ordZ = 1:ord(3)
    U = step(omg4,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
    aidx = aidx+nAngU;
    midx = midx+nMusU;
    R = blkdiag(eye(p),U);
    E  = R*Qz*E;
end

%% 
if ord(1) == 0 && ord(2) == 0 && ord(3) == 0
    Eu = E;
else
    Eu = upsample(E,[2 2 2],[1 2 3]);
end
H = Eu*d_M;

%% Extraction of filters
h = zeros(size(double(H),3),size(double(H),4),size(double(H),5),8);
for idx = 1:8
    h(:,:,:,idx) = H(idx,1,:,:,:);
end
f = flip(flip(flip(h,1),2),3);