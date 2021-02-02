% 範囲制約付き OCT ボリュームデータ復元（シミュレーション）
%
% Reference:
%
% S. Ono, "Primal-dual plug-and-play image restoration,"
% IEEE Signal Processing Letters, vol.24, no.8, pp.1108-1112, Aug. 2017.
%

% 可視化パラメータ
isVisible  = true;  % グラフ表示
isVerbose  = true;
slicePlane = 'YZ'; % 画像スライス方向
texture    = '2D';
obsScale   = 10;
estScale   = 20;

%
method = 'bm4d';

%% 最適化パラメータ
maxIter = 1e+3; % 最大繰返し回数
%isNoDcShk   = false; % DC成分閾値回避
%isEnvWght   = false; % 包絡線重み付け
% eta = 0, vmin=-Inf, vmax=Inf で ISTA相当
lambda  = 8e-4; %0.0032; % 正則化パラメータ 
vrange  = [-1.00 1.00]; % ハード制約
gamma1  = 1e-3; % ステップサイズ TODO 条件を確認
%

%% 観測パラメータ設定
wSigma = 1e-2; % ノイズ標準偏差
pScale = 1.00; % 光強度
pSigma = 4.00; % 広がり
pFreq  = 0.25; % 周波数

%% 合成辞書（パーセバルタイトフレーム）
import saivdr.dictionary.utility.*
fwdDic = IdentitySynthesisSystem();
adjDic = IdentityAnalysisSystem('IsVectorize',false);

%% ガウスノイズ除去
import saivdr.restoration.denoiser.*
gdnFcnG = GaussianDenoiserBm4d();

%% 原信号データ生成
depth   = 64; % 奥行
height  = depth; % 高さ
width   = 16; % 幅
phtm = phantom('Modified Shepp-Logan',depth);
sliceYZ = permute(phtm,[1 3 2]);
uSrc = 0.5*repmat(sliceYZ,[1 width 1]) + 1;
phi = RefractIdx2Reflect();
rSrc = phi.step(uSrc);

%% 原データ表示
if isVisible
    import saivdr.utility.*
    % 準備
    %fsqzline = @(x) squeeze(x(floor(height/2),floor(width/2),:));
    %
    hImg = figure(1);
    %
    vdvsrc = VolumetricDataVisualizer(...
        'Texture',texture,...
        'SlicePlane',slicePlane,...
        ...'DAspect',[1 1 3],...
        'VRange',[0 2]);
    subplot(2,3,1)
    vdvsrc.step(uSrc);
    xlabel(['Refract. Idx ' slicePlane ' slice'])
    %
    subplot(2,3,4)
    vdpsrc = VolumetricDataPlot(...
        'Direction','Z',...
        'NumPlots',2,...
        'Scales',[1 10]);
    vdpsrc.step(uSrc,rSrc);
    axis([0 size(uSrc,3) -1 2])
    legend('Refraction Idx','Reflectance x10','Location','best')
    title('Source')
end

%% 観測過程生成
msrProc = Coherence3(...
    'Scale',pScale,...
    'Sigma',pSigma,...
    'Frequency',pFreq);

pKernel = msrProc.Kernel;
if isVisible
    figure(2)
    plot(squeeze(pKernel))
    xlabel('Depth')
    ylabel('Intensity')
    title('Coherence function P')
end

%% 観測データ生成
rng(0)
vObs = msrProc.step(rSrc,'Forward') ...
    + wSigma*randn(size(rSrc));
% 反射率関数は非線形
if isVisible
    mymse = @(x,y) norm(x(:)-y(:),2)^2/numel(x);
    figure(hImg)
    subplot(2,3,2)
    vdvobs = VolumetricDataVisualizer(...
        'Texture',texture,....
        'SlicePlane',slicePlane,...
        ...'DAspect',[1 1 3],...
        'VRange',[-1 1],...
        'Scale',obsScale);    
    vdvobs.step(vObs);
    xlabel(sprintf('Obs %s slice: MSE = %6.4e',slicePlane,mymse(vObs,rSrc)))
    %
    subplot(2,3,5)
    vdpobs = VolumetricDataPlot(...
        'Direction','Z',...
        'NumPlots',2);
    vdpobs.step(vObs,0*vObs);
    axis([0 size(vObs,3) -1 2])
    legend('Observation','Estimation','Location','best')
    title('Observation')
end

%% 条件の表示
if isVerbose
    disp('-------------------------------')
    disp('データサイズ')
    disp('-------------------------------')
    disp(['幅×高×奥行　　　： ' num2str(width) 'x' num2str(height) 'x' num2str(depth)])
    disp('-------------------------------')
    disp('観測過程')
    disp('-------------------------------')
    disp(['光強度　　　　　　： ' num2str(pScale)])
    disp(['広がり　　　　　　： ' num2str(pSigma)])
    disp(['周波数　　　　　　： ' num2str(pFreq)])
    disp(['ノイズ標準偏差　　： ' num2str(wSigma)])
end


%% データ復元モニタリング準備
if isVisible
    vdv = VolumetricDataVisualizer(...
        'Texture',texture,...
        'SlicePlane',slicePlane,...
        ...'DAspect',[1 1 3],...
        'VRange',[-1 1],...
        'Scale',estScale);    
    r = vObs;
    %
    figure(hImg)
    subplot(2,3,3)
    vdv.step(r);
    hTitle3 = title(sprintf('Rfl Est(  0): MSE = %6.4e',...
        mymse(vObs,rSrc)));
    %
    subplot(2,3,6)
    vdp = VolumetricDataPlot(...
        'Direction','Z',...
        'NumPlots',1,...
        'Scales',10);
    vdp.step(r);
    axis([0 size(vObs,3) -1 2])
    legend('Reflectance x10','Location','best')
    title('Restoration')
end

%% 復元システム生成
pdpnp = PdPnPOct3(...
    'Observation',    vObs,...
    'Lambda',         lambda,...
    'Gamma1',         gamma1,...
    'VRange',         vrange,...
    'MeasureProcess', msrProc,...
    'Dictionary',     { fwdDic, adjDic },...
    'GaussianDenoiser', gdnFcnG );

disp(pdpnp)

%% 復元処理
for itr = 1:maxIter
    rEst = pdpnp.step();
    % Monitoring
    vdpobs.step(vObs,msrProc(rEst,'Forward'));
    vdp.step(rEst);
    vdv.step(rEst);
    set(hTitle3,'String',sprintf('Rfl Est (%3d): MSE = %6.4e',itr,mymse(rEst,rSrc)));
    %
    drawnow    
end

%% 結果表示
fprintf('Restoration of Reflection: MSE = %6.4e\n',mymse(rEst,rSrc));

%%
dt = char(datetime('now','TimeZone','local','Format','d-MM-y-HH-mm-ssZ'));
targetdir = ['./sim_clg_' dt];
if exist(targetdir,'dir') ~= 7
    mkdir(targetdir)
end
save([targetdir '/usrc_sim'],'uSrc')
save([targetdir '/rest_sim'],'rEst')
save([targetdir '/rsrc_sim'],'rSrc')
save([targetdir '/vobs_sim'],'vObs')

%% データ保存
options.method = method;
options.lambda = lambda;
options.gamma1 = gamma1;
options.vrange = vrange;
options.msrProc = msrProc;
options.fwdDic = fwdDic;
options.adjDic = adjDic;
options.gdnFcnG = gdnFcnG;
save([targetdir '/results_sim' ],'options') % TODO XML?