function [h,f,E] = fcn_type2d22c32o33(angs,mus,isnodc)
%
% $Id: fcn_type2d22c32o33.m 850 2015-10-29 21:19:55Z sho $
% 
ps = 3;
pa = 2;
pn = min(ps,pa);
px = max(ps,pa);
nCh = ps + pa;

if nargin < 3
    isnodc = false;
end

if nargin < 2
    mus  = [ -1 1   -1 1    1 -1  -1 1 1     1 1 1  -1 -1  1 1 1  -1 -1 ].'; 
end

if nargin < 1
    angs = [ -pi/2  -pi/2  -pi/2  -pi/2 0 0  0 0 0   0     0 0 0   0    ].';
end

ord = [3 3];

%% Preperation
import saivdr.dictionary.utility.PolyPhaseMatrix2d
import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
import saivdr.dictionary.utility.OrthonormalMatrixFactorizationSystem

omgW = OrthonormalMatrixGenerationSystem();
omgU = OrthonormalMatrixGenerationSystem();

%% Delay chain
clear d_M
d_M(1,1,1,1) = 1;
d_M(2,1,2,1) = 1;
d_M(3,1,1,2) = 1;
d_M(4,1,2,2) = 1;
d_M = PolyPhaseMatrix2d(d_M);

%% Butterfly

Z = zeros(pn,abs(ps-pa));
I = eye(abs(ps-pa));
B = [ eye(pn) Z eye(pn) ; Z.' sqrt(2)*I Z.' ; eye(pn) Z -eye(pn) ]/sqrt(2);
  
%% Lambda

clear Loy;
Loy(:,:,1,1) = [ eye(pn) zeros(pn,px) ; zeros(px,pn) zeros(px) ]; 
Loy(:,:,2,1) = [ zeros(pn) zeros(pn,px) ; zeros(px,pn) eye(px) ]; 
Loy = PolyPhaseMatrix2d(Loy);

clear Ley;
Ley(:,:,1,1) = [ eye(px) zeros(px,pn) ; zeros(pn,px) zeros(pn) ]; 
Ley(:,:,2,1) = [ zeros(px) zeros(px,pn) ; zeros(pn,px) eye(pn) ]; 
Ley = PolyPhaseMatrix2d(Ley);

clear Lox;
Lox(:,:,1,1) = [ eye(pn) zeros(pn,px) ; zeros(px,pn) zeros(px) ]; 
Lox(:,:,1,2) = [ zeros(pn) zeros(pn,px) ; zeros(px,pn) eye(px) ]; 
Lox = PolyPhaseMatrix2d(Lox);

clear Lex;
Lex(:,:,1,1) = [ eye(px) zeros(px,pn) ; zeros(pn,px) zeros(pn) ]; 
Lex(:,:,1,2) = [ zeros(px) zeros(px,pn) ; zeros(pn,px) eye(pn) ]; 
Lex = PolyPhaseMatrix2d(Lex);

%% Qd
Qoy = B*Loy*B;
Qey = B*Ley*B;
Qox = B*Lox*B;
Qex = B*Lex*B;

%% Constraction
nAngW = 3;
nAngU = 1;
nMusW = 3;
nMusU = 2;
aidx  = 4;
midx  = 7;

%% E0
angs_(1:3) = angs(1:3);
angs_(4)   = 0;
mus_(1:6)  = mus(1:6);
mus_(7)    = 1;
mus_(8)    = 1;
[~,~,E0] = eznsolt.fcn_type1d22c22([1 1],angs_,mus_,false);
W0 = step(omgW,angs(aidx:aidx+nAngW-1),mus(midx:midx+nMusW-1));
aidx = aidx+nAngW;
midx = midx+nMusW;
%
R0 = blkdiag(W0(:,1:pn),eye(pn));
E  = R0*E0;

%% Vertical extension
for ordY = 1:(ord(1)-1)/2
    W = step(omgW,angs(aidx:aidx+nAngW-1),mus(midx:midx+nMusW-1));
    aidx = aidx+nAngW;
    midx = midx+nMusW;
    U = step(omgU,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
    aidx = aidx+nAngU;
    midx = midx+nMusU;    
    %
    Ro = blkdiag(eye(ps),U);
    Re = blkdiag(W,eye(pa));
    E = Re*Qey*Ro*Qoy*E;
end

if isnodc
     omf = OrthonormalMatrixFactorizationSystem();
     angsi(1) = angs(1);
     angsi(2) = 0;
     angsi(3) = 0;
     musi(1)  = mus(1);
     musi(2)  = mus(2);
     musi(3)  = 1;
     Wi  = step(omgW,angsi,musi);
     [ang_,mus_] = step(omf,(W*W0*Wi).');
     angs(aidx:aidx+1) = ang_(1:2);
     mus(midx) = mus_(1);
end

%% Horizontal extension
for ordX = 1:(ord(2)-1)/2
    W = step(omgW,angs(aidx:aidx+nAngW-1),mus(midx:midx+nMusW-1));
    aidx = aidx+nAngW;
    midx = midx+nMusW;
    U = step(omgU,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
    aidx = aidx+nAngU;
    midx = midx+nMusU;    
    %
    Ro = blkdiag(eye(ps),U);
    Re = blkdiag(W,eye(pa));
    E = Re*Qex*Ro*Qox*E;
end

%% 
Eu = upsample(E,[2 2],[1 2]);
H = Eu*d_M;

%% Extraction of filters
h = zeros(size(double(H),3),size(double(H),4),nCh);
for idx = 1:nCh
    h(:,:,idx) = H(idx,1,:,:);
end
f = flip(flip(h,1),2);