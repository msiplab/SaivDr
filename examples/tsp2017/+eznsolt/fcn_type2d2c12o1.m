function [h,f,E] = fcn_type2d2c12o1(angs,mus,isnodc)
%
% $Id: fcn_type2d2c12o1.m 850 2015-10-29 21:19:55Z sho $
% 
if nargin < 3
    isnodc = false;
end

if nargin < 2 
    mus  = [ 1 1 1 1 -1 ].'; 
end

if nargin < 1
    angs = 0;
end

%% Preperation
import saivdr.dictionary.utility.PolyPhaseMatrix1d
import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem

omgU = OrthonormalMatrixGenerationSystem();

%% Delay chain
clear d_M
d_M(1,1,1) = 1;
d_M(2,1,2) = 1;
d_M = PolyPhaseMatrix1d(d_M);

%% Constraction
nMusW = 1;
nAngU = 1;
nMusU = 2;
aidx  = 1;
midx  = 3;

%% E0
if isnodc
    mus(1)     = 1;
    mus(midx)  = 1;
end
mus_(1:2)  = mus(1:2);
mus_(3)    = 1;
[~,~,E0] = eznsolt.fcn_type1d2c11(1,mus_,false);
W0 = mus(midx:midx+nMusW-1);
midx = midx + nMusW;
U0 = step(omgU,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
%
R0 = blkdiag(W0(:,1),U0(:,1));
E  = R0*E0;

%%
Eu = upsample(E,2);
H = Eu*d_M;

%% Extraction of filters
h = permute(double(H),[3 1 2]);
f = flip(h,1);