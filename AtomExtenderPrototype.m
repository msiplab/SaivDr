% initialize
nBlks = 6;

obb1=saivdr.dictionary.nsoltx.mexsrcs.Order1BuildingBlockTypeI();
omgV = saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem();
omgG = saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem();
Edft = complex(zeros(4));
for tu = 0:3
    for tx =0:3
        n = rem(tu*(2*tx+1),2*4);
        Edft(tu+1,tx+1) = exp(-1i*pi*n/4)/sqrt(4);
    end
end
V0 = step(omgV,2*pi*rand(6,1),ones(4,1));
W1 = step(omgG,2*pi*rand(1,1),ones(2,1));
U1 = step(omgG,2*pi*rand(1,1),ones(2,1));
angB1 = 2*pi*rand(1,1);
W2 = step(omgG,2*pi*rand(1,1),ones(2,1));
U2 = step(omgG,2*pi*rand(1,1),ones(2,1));
angB2 = 2*pi*rand(1,1);

C1 = [-1j*cos(angB1),-1j*sin(angB1);cos(angB1),-sin(angB1)];
S1 = 1j*[-1j*sin(angB1),-1j*cos(angB1);sin(angB1),-cos(angB1)];
Bh1 = [C1 ,conj(C1);S1, conj(S1)]/sqrt(2);
C2 = [-1j*cos(angB2),-1j*sin(angB2);cos(angB2),-sin(angB2)];
S2 = 1j*[-1j*sin(angB2),-1j*cos(angB2);sin(angB2),-cos(angB2)];
Bh2 = [C2 ,conj(C2);S2, conj(S2)]/sqrt(2);

%lattice
x=(rand(4,nBlks)+1i*rand(4,nBlks));

%xs=repmat(x(:),3,1);
xs = x(:);

tmp0 = V0*conj(Edft)*x;
tmp1 = Bh1.'*tmp0;
tmp2 = tmp1;
p = tmp2(1:2,end);
tmp2(1:2,2:end) = tmp2(1:2,1:end-1);
tmp2(1:2,1) = p;
% p = tmp2(3:4,1);
% tmp2(3:4,1:end-1) = tmp6(3:4,2:end);
% tmp2(3:4,end)= p;%TODO
tmp3=conj(Bh1)*tmp2;
tmp4=blkdiag(W1,U1)*tmp3;

tmp5=Bh2.'*tmp4;
tmp6=tmp5;
q = tmp6(3:4,1);
tmp6(3:4,1:end-1) = tmp6(3:4,2:end);
tmp6(3:4,end)= q;
% q = tmp6(1:2,end);
% tmp6(1:2,2:end) = tmp6(1:2,1:end-1);
% tmp6(1:2,1) = q;
tmp7=conj(Bh2)*tmp6;
tmp8=blkdiag(W2,U2)*tmp7;
yl = tmp8;

%filter array
E0 = V0*Edft;
E1 = step(obb1,E0,W1,U1,angB1,2,4);
E2 = step(obb1,E1,W2,U2,angB2,2,4);
yf = zeros(4,length(xs)/4);
for idx=1:4
    yf(idx,:) = downsample(circshift(cconv(E2(idx,:).',xs,length(xs)),-4),4,3).';
end

yf
yl

err = sum(abs(yf(:)-yl(:)))