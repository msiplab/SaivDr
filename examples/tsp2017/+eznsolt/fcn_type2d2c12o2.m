function [h,f,E] = fcn_type2d2c12o2(angs,mus,isnodc)
%
% $Id: fcn_type2d2c12o2.m 850 2015-10-29 21:19:55Z sho $
% 
ps = 1;
pa = 2;
pn = min(ps,pa);
px = max(ps,pa);

if nargin < 3
    isnodc = false;
end

if nargin < 2
    mus  = [ -1 1 1 -1 -1 -1 ];
end

if nargin < 1
    angs = [ 0 0 ];
end

%% Preperation
import saivdr.dictionary.utility.PolyPhaseMatrix1d
import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
import saivdr.dictionary.utility.OrthonormalMatrixFactorizationSystem

omgU = OrthonormalMatrixGenerationSystem();

%% Delay chain
clear d_M
d_M(1,1,1) = 1;
d_M(2,1,2) = 1;
d_M = PolyPhaseMatrix1d(d_M);

%% 1-D DCT
C_MJ_M = ...
    [ 1 1 ;
    -1 1 ]/sqrt(2);

%% Butterfly
Z = zeros(pn,abs(ps-pa));
I = eye(abs(ps-pa));
B = [ eye(pn) Z eye(pn) ; Z.' sqrt(2)*I Z.' ; eye(pn) Z -eye(pn) ]/sqrt(2);
  
%% Lambda
clear Lo
Lo(:,:,1) = [ eye(pn) zeros(pn,px) ; zeros(px,pn) zeros(px) ]; 
Lo(:,:,2) = [ zeros(pn) zeros(pn,px) ; zeros(px,pn) eye(px) ]; 
Lo = PolyPhaseMatrix1d(Lo);

clear Le
Le(:,:,1) = [ eye(px) zeros(px,pn) ; zeros(pn,px) zeros(pn) ]; 
Le(:,:,2) = [ zeros(px) zeros(px,pn) ; zeros(pn,px) eye(pn) ]; 
Le = PolyPhaseMatrix1d(Le);

%% Qd
Qo = B*Lo*B;
Qe = B*Le*B;

%% Constraction
nMusW = 1;
nAngU = 1;
nMusU = 2;
aidx=1;
midx=1;

%% E0
W0  = mus(midx:midx+nMusW-1);
midx = midx+nMusW;
%
U0  = step(omgU,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
aidx = aidx+nAngU;
midx = midx+nMusU;
%
R0 = blkdiag(W0(:,1),U0(:,1));
E = R0*C_MJ_M;

%% Order extension
if isnodc
    mus(midx) = mus(1);
end
W = mus(midx:midx+nMusW-1);
midx = midx+nMusW;
U = step(omgU,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
%
Ro = blkdiag(W,eye(pa));
Re = blkdiag(eye(ps),U);
E = Re*Qe*Ro*Qo*E;

%% 
Eu = upsample(E,2);
H = Eu*d_M;

%% Extraction of filters
h = permute(double(H),[3 1 2]);
f = flip(h,1);