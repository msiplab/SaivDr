% 観測過程Φの近似の確認
close all

%% 屈折率の値の範囲
vmin = 1.0;
vmax = 1.5;

%% 準備
vrange = [vmin vmax];
u = (vmax-vmin)*rand(8,8,16)+vmin;
v = (vmax-vmin)*rand(8,8,16)+vmin;

%% 反射率計算
fr = @(x,y) abs(y-x)*(y-x)/(y+x)^2;
f1 = @(x,y) abs(vmax-vmin)*(y-x)/(vmax+vmin)^2;
f2 = @(x,y) abs(y-x)*(y-x)/(vmax+vmin)^2;

%% グラフ表示
subplot(2,3,1), fmesh(fr,[vmin vmax vmin vmax])
title('Reflection \phi')
xlabel('n_1')
ylabel('n_2')
subplot(2,3,2), fmesh(f1,[vmin vmax vmin vmax])
title('Linear Aprx. \phi_1')
subplot(2,3,3), fmesh(f2,[vmin vmax vmin vmax])
title('Signed-Quadratic Aprx. \phi_2')

subplot(2,3,5), fmesh(@(x,y) fr(x,y)-f1(x,y),[vmin vmax vmin vmax])
axis([vmin vmax vmin vmax -2e-2 2e-2])
title('Difference \phi-\phi_1')
subplot(2,3,6), fmesh(@(x,y) fr(x,y)-f2(x,y),[vmin vmax vmin vmax])
title('Difference \phi-\phi_2')
axis([vmin vmax vmin vmax -2e-2 2e-2])

%% Φの近似評価
u = (vmax-vmin)*rand(8,8,16)+vmin;
kernelxy = kron([ 1 2 1 ].', [ 1 2 1 ]);
kernelz  = permute([ 1 0 -1 ].',[ 2 3 1 ]);
sobel3d = convn(kernelxy,kernelz)/32;

% Φ
r2r0 = RefractIdx2Reflect('PhiMode','Reflection');
r0 = r2r0.step(u);
% Φ1
r2r1 = RefractIdx2Reflect('PhiMode','Linear','VRange',[vmin vmax]);
r1 = r2r1.step(u);
% Φ2
r2r2 = RefractIdx2Reflect('PhiMode','Signed-Quadratic','VRange',[vmin vmax]);
r2 = r2r2.step(u);

% Φ-Φ1
disp('Φ1の評価')
disp(norm(r0(:)-r1(:))^2)
% Φ-Φ2
disp('Φ2の評価')
disp(norm(r0(:)-r2(:))^2)

%% グラフ表示
figure(2)
subplot(1,2,1), fmesh(fr,[vmin vmax vmin vmax])
title('Reflection \phi')
daspect([2 2 1])
xlabel('n_1')
ylabel('n_2')

subplot(1,2,2), fmesh(f1,[vmin vmax vmin vmax])
title('Linear Aprx. \phi_1')
daspect([2 2 1])
xlabel('n_1')
ylabel('n_2')