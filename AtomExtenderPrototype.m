% initialize
nBlks = 4;
nDec = 1;
nCh = 5;
gCh = ceil(nCh/2);
lCh = floor(nCh/2);

omgV = saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem();
omgG = saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem();
omgL = saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem();
Edft = complex(zeros(nDec));
for tu = 0:nDec-1
    for tx = 0:nDec-1
        n = rem(tu*(2*tx+1),2*nDec);
        Edft(tu+1,tx+1) = exp(-1i*pi*n/nDec)/sqrt(nDec);
    end
end

lm=[eye(nDec);zeros(nCh-nDec,nDec)];

npv = nCh*(nCh-1)/2;
npg = gCh*(gCh-1)/2;
npl = lCh*(lCh-1)/2;
npb = floor(nCh/4);

angs = 2*pi*rand(npv+2*(npg+npl+npb),1);

idx = 0;
V0 = step(omgV,angs(idx+1:idx+npv),ones(nCh,1));
idx = idx + npv;
W1 = step(omgL,angs(idx+1:idx+npl),ones(lCh,1));
idx = idx + npl;
U1 = step(omgL,angs(idx+1:idx+npl),ones(lCh,1));
idx = idx + npl;
angB1 = angs(idx+1:idx+npb);
idx = idx + npb;
W2 = step(omgG,angs(idx+1:idx+npg),ones(gCh,1));
%W2 = eye(gCh);
idx = idx + npg;
U2 = step(omgG,angs(idx+1:idx+npg),ones(gCh,1));
%U2 = eye(gCh);
idx = idx + npg;
angB2 = angs(idx+1:idx+npb);
%angB2 = pi/4*ones(npb,1);

C1 = [-1j*cos(angB1),-1j*sin(angB1);cos(angB1),-sin(angB1)];
S1 = 1j*[-1j*sin(angB1),-1j*cos(angB1);sin(angB1),-cos(angB1)];
Bh1 = [C1 ,conj(C1), zeros(lCh,1);S1, conj(S1), zeros(lCh,1); zeros(1,2*lCh),sqrt(2)]/sqrt(2);
C2 = [-1j*cos(angB2),-1j*sin(angB2);cos(angB2),-sin(angB2)];
S2 = 1j*[-1j*sin(angB2),-1j*cos(angB2);sin(angB2),-cos(angB2)];
Bh2 = [C2 ,conj(C2), zeros(lCh,1); S2, conj(S2), zeros(lCh,1); zeros(1,2*lCh),sqrt(2)]/sqrt(2);

%lattice
x=(rand(nDec,nBlks)+1i*rand(nDec,nBlks));
xs = x(:);
% xs = srcSeq.';
% x = reshape(xs,nDec,nBlks);
cBh1 = conj(Bh1);

tmp0 = V0*lm*conj(Edft)*x;
tmp1 = cBh1'*tmp0;
tmp2 = tmp1;
% p = tmp2(1:2,end);
% tmp2(1:2,2:end) = tmp2(1:2,1:end-1);
% tmp2(1:2,1) = p;
p = tmp2(lCh+1:end-1,1);
tmp2(lCh+1:end-1,1:end-1) = tmp2(lCh+1:end-1,2:end);
tmp2(lCh+1:end-1,end)= p;%TODO
tmp3=cBh1*tmp2;
tmp4=blkdiag(W1,U1,1)*tmp3;

cBh2 = conj(Bh2);
tmp5=cBh2'*tmp4;
tmp6=tmp5;
% q = tmp6(3:4,1);
% tmp6(3:4,1:end-1) = tmp6(3:4,2:end);
% tmp6(3:4,end)= q;
q = tmp6(1:lCh,end);
tmp6(1:lCh,2:end) = tmp6(1:lCh,1:end-1);
tmp6(1:lCh,1) = q;
tmp7=cBh2*tmp6;
tmp8=blkdiag(W2,eye(lCh))*blkdiag(eye(lCh),U2)*tmp7;
yl = tmp8;

%filter array
% obb1=saivdr.dictionary.nsoltx.mexsrcs.Order1BuildingBlockTypeI();
E0 = V0*lm*Edft;
% E1 = step(obb1,E0,W1,U1,angB1,lCh,nDec);
% E2 = step(obb1,E1,W2,U2,angB2,lCh,nDec);
obb2 = saivdr.dictionary.nsoltx.mexsrcs.Order2BuildingBlockTypeII();
E2 = step(obb2,E0,W1, U1, angB1, W2, U2, angB2, lCh, nDec);
yf = zeros(4,length(xs)/nDec);
for idx=1:nCh
    yf(idx,:) = downsample(circshift(cconv(xs,E2(idx,:).',length(xs)),-nDec),nDec,nDec-1).';
end

yf
yl

err = sum(abs(yf(:)-yl(:)))