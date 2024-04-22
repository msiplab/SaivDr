function [h,f,E] = fcn_type1d2c11(ord,mus,isnodc)
%
% $Id: fcn_type1d2c11.m 850 2015-10-29 21:19:55Z sho $
% 
if nargin < 3
    isnodc = false;
end

if nargin < 2
    mus  = [ 1 1 1 ];
end

if nargin < 1
    ord = 1;
end

%% Preperation
import saivdr.dictionary.utility.PolyPhaseMatrix1d

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
B = [ 1 1 ; 1 -1 ]/sqrt(2);
  
%% Lambda
clear L
L(:,:,1) = [ 1 0 ; 0 0 ]; 
L(:,:,2) = [ 0 0 ; 0 1 ];
L = PolyPhaseMatrix1d(L);

%% Qd
Q = B*L*B;

%% Constraction
nMusW = 1;
nMusU = 1;
midx=1;

%% E0
if isnodc
    mus(1)  = 1;
end
W0  = mus(midx:midx+nMusW-1);
midx = midx+nMusW;
%
U0  = mus(midx:midx+nMusU-1);
midx = midx+nMusU;
%
R0 = blkdiag(W0(:,1),U0(:,1));
E = R0*C_MJ_M;

%% Extension
for iOrd = 1:ord
    U = mus(midx:midx+nMusU-1);
    midx = midx+nMusU;
    R = blkdiag(1,U);
    E  = R*Q*E;
end

%% 
if ord == 0
    Eu = E;
else
    Eu = upsample(E,2);
end
H = Eu*d_M;

%% Extraction of filters
h = permute(double(H),[3 1 2]);
f = flip(h,1);