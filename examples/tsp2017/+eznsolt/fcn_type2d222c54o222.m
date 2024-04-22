function [h,f,E] = fcn_type2d222c54o222(angs,mus,isnodc)
%
% $Id: fcn_type2d222c54o222.m 850 2015-10-29 21:19:55Z sho $
% 
ps = 5;
pa = 4;
pn = min(ps,pa);
px = max(ps,pa);
nChs = ps + pa;

if nargin < 3
    isnodc = false;
end

if nargin < 2
    mus  = repmat([ones(ps,1); -ones(pa,1)],[4 1]);
end

if nargin < 1
    angs = zeros(16*4,1);
end

ord = [2 2 2];

%% Preperation
import saivdr.dictionary.utility.PolyPhaseMatrix3d
import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
import saivdr.dictionary.utility.OrthonormalMatrixFactorizationSystem

omgW = OrthonormalMatrixGenerationSystem();
omgU = OrthonormalMatrixGenerationSystem();

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

Z = zeros(pn,abs(ps-pa));
I = eye(abs(ps-pa));
B = [ eye(pn) Z eye(pn) ; Z.' sqrt(2)*I Z.' ; eye(pn) Z -eye(pn) ]/sqrt(2);
  
%% Lambda
clear Loy;
Loy(:,:,1,1,1) = [ eye(pn) zeros(pn,px) ; zeros(px,pn) zeros(px) ]; 
Loy(:,:,2,1,1) = [ zeros(pn) zeros(pn,px) ; zeros(px,pn) eye(px) ]; 
Loy = PolyPhaseMatrix3d(Loy);

clear Ley;
Ley(:,:,1,1,1) = [ eye(px) zeros(px,pn) ; zeros(pn,px) zeros(pn) ]; 
Ley(:,:,2,1,1) = [ zeros(px) zeros(px,pn) ; zeros(pn,px) eye(pn) ]; 
Ley = PolyPhaseMatrix3d(Ley);

clear Lox;
Lox(:,:,1,1,1) = [ eye(pn) zeros(pn,px) ; zeros(px,pn) zeros(px) ]; 
Lox(:,:,1,2,1) = [ zeros(pn) zeros(pn,px) ; zeros(px,pn) eye(px) ]; 
Lox = PolyPhaseMatrix3d(Lox);

clear Lex;
Lex(:,:,1,1,1) = [ eye(px) zeros(px,pn) ; zeros(pn,px) zeros(pn) ]; 
Lex(:,:,1,2,1) = [ zeros(px) zeros(px,pn) ; zeros(pn,px) eye(pn) ]; 
Lex = PolyPhaseMatrix3d(Lex);

clear Loz;
Loz(:,:,1,1,1) = [ eye(pn) zeros(pn,px) ; zeros(px,pn) zeros(px) ]; 
Loz(:,:,1,1,2) = [ zeros(pn) zeros(pn,px) ; zeros(px,pn) eye(px) ]; 
Loz = PolyPhaseMatrix3d(Loz);

clear Lez;
Lez(:,:,1,1,1) = [ eye(px) zeros(px,pn) ; zeros(pn,px) zeros(pn) ]; 
Lez(:,:,1,1,2) = [ zeros(px) zeros(px,pn) ; zeros(pn,px) eye(pn) ]; 
Lez = PolyPhaseMatrix3d(Lez);

%% Qd
Qoy = B*Loy*B;
Qey = B*Ley*B;
Qox = B*Lox*B;
Qex = B*Lex*B;
Qoz = B*Loz*B;
Qez = B*Lez*B;

%% Constraction
nAngW = ps*(ps-1)/2;
nAngU = pa*(pa-1)/2;
nMusW = ps;
nMusU = pa;
aidx=1;
midx=1;

%% E0
W0  = step(omgW,angs(aidx:aidx+nAngW-1),mus(midx:midx+nMusW-1));
aidx = aidx+nAngW;
midx = midx+nMusW;
%
U0  = step(omgU,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
aidx = aidx+nAngU;
midx = midx+nMusU;
%
R0 = blkdiag(W0(:,1:4),U0(:,1:4));
E = R0*C_MJ_M;

%% Vertical extension
for ordY = 1:ord(1)/2
    Wy = step(omgW,angs(aidx:aidx+nAngW-1),mus(midx:midx+nMusW-1));
    aidx = aidx+nAngW;
    midx = midx+nMusW;
    U = step(omgU,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
    aidx = aidx+nAngU;
    midx = midx+nMusU;    
    %
    Ro = blkdiag(eye(ps),U);
    Re = blkdiag(Wy,eye(pa));
    E = Re*Qey*Ro*Qoy*E;
end

%% Horizontal extension
for ordX = 1:ord(2)/2
    Wx = step(omgW,angs(aidx:aidx+nAngW-1),mus(midx:midx+nMusW-1));
    aidx = aidx+nAngW;
    midx = midx+nMusW;
    U = step(omgU,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
    aidx = aidx+nAngU;
    midx = midx+nMusU;    
    %
    Ro = blkdiag(eye(ps),U);
    Re = blkdiag(Wx,eye(pa));
    E = Re*Qex*Ro*Qox*E;
end

if isnodc
    omf = OrthonormalMatrixFactorizationSystem();
    [ang_,mus_] = step(omf,(Wx*Wy*W0).');
    angs(aidx:aidx+nAngW-2) = ang_(1:nAngW-1);
    mus(midx)  = mus_(1);
end

%% Depth extension
for ordZ = 1:ord(3)/2
    Wz = step(omgW,angs(aidx:aidx+nAngW-1),mus(midx:midx+nMusW-1));
    aidx = aidx+nAngW;
    midx = midx+nMusW;
    U = step(omgU,angs(aidx:aidx+nAngU-1),mus(midx:midx+nMusU-1));
    aidx = aidx+nAngU;
    midx = midx+nMusU;    
    %
    Ro = blkdiag(eye(ps),U);
    Re = blkdiag(Wz,eye(pa));
    E = Re*Qez*Ro*Qoz*E;
end

%%
if ord(1) == 0 && ord(2) == 0 && ord(3) == 0
    Eu = E;
else
    Eu = upsample(E,[2 2 2],[1 2 3]);
end
H = Eu*d_M;

%% Extraction of filters
h = zeros(size(double(H),3),size(double(H),4),size(double(H),5),nChs);
for idx = 1:nChs
    h(:,:,:,idx) = H(idx,1,:,:,:);
end
f = flip(flip(flip(h,1),2),3);