% 範囲制約付き OCT ボリュームデータ復元（シミュレーション）
%
% Reference:
%
% S. Ono, "Primal-dual plug-and-play image restoration,"
% IEEE Signal Processing Letters, vol.24, no.8, pp.1108-1112, Aug. 2017.
%

%% 並列プール設定
%%{
%poolobj = gcp('nocreate');
%delete(poolobj);
%nWorkers = 2;
%parpool(nWorkers)

%%  GPU 設定
%spmd
%  gpuDevice( 1 + mod( labindex - 1, gpuDeviceCount ) )
%end
%%}

%% 変換パラメータ
nLevels = 1; % ツリーレベル
splitfactor = [1 1 1]; %2*ones(1,3);  % 垂直・水平・奥行方向並列度
padsize = 2^(nLevels-1)*ones(1,3); % OLS/OLA パッドサイズ
isintegritytest = false; % 整合性テスト
useparallel = false; % 並列化
usegpu = true; % GPU
issingle = true; % 単精度
method = 'udht';
%method = 'bm4d';
isIsta      = false; % ISTA
isNoDcShk   = false;  % DC成分閾値回避
%isEnvWght   = false;  % 包絡線重み付け

%% 最大繰り返し回数
maxIter = 1000;

%% 可視化パラメータ
isVerbose  = true;
isVisible  = true;  % グラフ表示
monint     = 10;    % モニタリング間隔
texture    = '2D';
slicePlane = 'YZ';  % 画像スライス方向
daspect    = [1 1 3];
obsScale =  20; % 観測データ用モニタの輝度調整
estScale = 200; % 復元データ用モニタの輝度調整
vdpScale = [1 10]; % プロット用スケール調整

%% 最適化パラメータ
% eta = 0, vmin=-Inf, vmax=Inf で ISTA相当
sizecomp = true; % 正則化パラメータのサイズ補正
if strcmp(method,'udht')
    barlambda  = 1e-8; % 正則化パラメータ (変換係数のスパース性）
    bareta     = 1e-8; % 正則化パラメータ（奥行方向の全変動）
elseif strcmp(method,'bm4d')
    barlambda  = 1e-8; % 正則化パラメータ (変換係数のスパース性）
    bareta     = 1e-8; % 正則化パラメータ（奥行方向の全変動）
else
    error('METHOD')
end
vrange  = [1.00 1.50]; % ハード制約
gamma1  = 1e-3; % ステップサイズ TODO 条件を確認
% 観測モデル（復元処理のみ）
% PHIMODE in { 'Linear', 'Signed-Quadratic', 'Reflection' }
phiMode   = 'Linear';
%phiMode   = 'Signed-Quadratic'; % γ1を小さく 1e-7
%phiMode   = 'Reflection'; 　　　% γ1を小さく 1e-30
%
% TODO 非線形モデルは勾配消失問題にあたるよう。γ1の制御が必要か。
%

%% 観測パラメータ設定
wSigma = 1e-2; % ノイズ標準偏差
pScale = 1.00; % 光強度
pSigma = 4.00; % 広がり
pFreq  = 0.25; % 周波数

%% 合成辞書（パーセバルタイトフレーム）＆ ガウスノイズ除去
if strcmp(method,'udht')
    import saivdr.dictionary.udhaar.*
    import saivdr.restoration.denoiser.*
    fwdDic = UdHaarSynthesis3dSystem();
    adjDic = UdHaarAnalysis3dSystem('NumberOfLevels',nLevels);
    gdnFcnG = GaussianDenoiserSfth();
    gdnFcnH = GaussianDenoiserSfth();
elseif strcmp(method,'bm4d')
    import saivdr.dictionary.utility.*
    import saivdr.restoration.denoiser.*    
    fwdDic = IdentitySynthesisSystem();
    adjDic = IdentityAnalysisSystem('IsVectorize',false);
    gdnFcnG = GaussianDenoiserBm4d();
    gdnFcnH = GaussianDenoiserSfth();
else
    error('METHOD')
end

%% 原信号データ生成
depth   = 64; % 奥行
height  = depth; % 高さ
width   = 16; % 幅
phtm = phantom('Modified Shepp-Logan',depth);
sliceYZ = permute(phtm,[1 3 2]);
uSrc = 0.5*repmat(sliceYZ,[1 width 1]) + 1;

%% 原データ表示
if isVisible
    import saivdr.utility.*    
    phi = RefractIdx2Reflect();
    % 準備
    %fsqzline = @(x) squeeze(x(floor(height/2),floor(width/2),:));
    %
    hImg = figure(1);
    %
    vdvsrc = VolumetricDataVisualizer(...
        'Texture',texture,...
        'VRange',[0 2]);
    if strcmp(texture,'2D')
        vdvsrc.SlicePlane = slicePlane;
    else
        vdvsrc.DAspect = daspect;
    end
        
    subplot(2,3,1)
    vdvsrc.step(uSrc);
    xlabel(['Refract. Idx ' slicePlane ' slice'])
    %
    subplot(2,3,4)
    vdpsrc = VolumetricDataPlot(...
        'Direction','Z',...
        'NumPlots',2,...
        'Scales',vdpScale);
    vdpsrc.step(uSrc,phi.step(uSrc));
    axis([0 size(uSrc,3) -1 2])
    legend('Refraction Idx',['Reflectance x' num2str(vdpScale(2))],'Location','best')
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
vObs = msrProc.step(phi.step(uSrc),'Forward') ...
    + wSigma*randn(size(uSrc));
% 反射率関数は非線形
if isVisible
    mymse = @(x,y) norm(x(:)-y(:),2)^2/numel(x);
    figure(hImg)
    subplot(2,3,2)
    rSrc = phi.step(uSrc);
    vdvobs = VolumetricDataVisualizer(...
        'Texture',texture,....
        'VRange',[-1 1],...
        'Scale',obsScale);    
    if strcmp(texture,'2D')
        vdvobs.SlicePlane = slicePlane;
    else
        vdvobs.DAspect = daspect;
    end
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

%% Z方向包絡線検出 → η設定
%{
if isEnvWght
    % フィルタリング
    v = msrProc(msrProc(vObs,'Adjoint'),'Forward');
    
    % Z方向包絡線抽出
    env = fcn_env_z(v)/(sum(abs(pKernel(:))))^2;
    
    % 平滑化
    env = imgaussfilt3(env,2);
    
    % 正則化パラメータη重み生成
    emax = max(env(:));
    emin = min(env(:));
    escales = ((emax-env)/(emax-emin));
    mscales = mean(escales(:));
    options.eta = options.eta * escales/mscales;
    if isVisible
        figure(3)
        widthV  = size(vObs,2);
        heightV = size(vObs,1);
        plot(squeeze(vObs(heightV/2,widthV/2,:)))
        xlabel('Depth')
        ylabel('Intensity')
        hold on
        plot(squeeze(env(heightV/2,widthV/2,:)))
        plot(squeeze(escales(heightV/2,widthV/2,:)))
        legend('Observation','Envelope after P''P','Weight for \eta','best')
        title('Z-direction sequence a the vertical and horizontal center.')
        hold off
        
        figure(4)
        imshow(squeeze(escales(:,widthV/2,:)))
        title('Weight map for \eta. Y-Z slice at the horizontal center. ')
    end
end
%}

%% データ復元モニタリング準備
if isVisible
    vdv = VolumetricDataVisualizer(...
        'Texture',texture,...
        'VRange',[-1 1],...
        'Scale',estScale);    
    if strcmp(texture,'2D')
       vdv.SlicePlane = slicePlane;
    else
        vdv.DAspect = daspect;
    end        
    phiapx = RefractIdx2Reflect(...
        'PhiMode',phiMode,...
        'VRange', vrange);
    r = phiapx.step(vObs);
    %
    figure(hImg)
    subplot(2,3,3)
    vdv.step(r);
    hTitle3 = title(sprintf('Rfl Est(  0): MSE = %6.4e',...
        mymse(vObs,phi.step(uSrc))));
    %
    subplot(2,3,6)
    vdp = VolumetricDataPlot(...
        'Direction','Z',...
        'NumPlots',2,...
        'Scales',vdpScale);
    vdp.step(vObs,r);
    axis([0 size(vObs,3) -1 2])
    legend('Refraction Idx','Reflectance x10','Location','best')
    title('Restoration')
end

%% 復元システム生成
pdshshc = PdsHsHcOct3(...
    'Observation',    vObs,...
    'Lambda',         barlambda,...
    'Eta',            bareta,...
    'IsSizeCompensation', sizecomp,...
    'Gamma1',         gamma1,...
    'PhiMode',        phiMode,...
    'VRange',         vrange,...
    'MeasureProcess', msrProc,...
    'Dictionary',     { fwdDic, adjDic },...
    'GaussianDenoiser', { gdnFcnG, gdnFcnH },...
    'SplitFactor',    splitfactor,...
    'PadSize',        padsize,...
    'UseParallel',    useparallel,...
    'UseGpu',         usegpu,...
    'IsIntegrityTest', isintegritytest);

disp(pdshshc)

%% 復元処理
for itr = 1:maxIter
    tic
    uEst = pdshshc.step();
    if isVerbose && itr==1
        lambda = pdshshc.LambdaCompensated;
        eta   = pdshshc.EtaCompensated;
        disp(['lambda = ' num2str(lambda)]);
        disp(['eta    = ' num2str(eta)]);
    end
    % Monitoring
    if isVisible && (itr==1 || mod(itr,monint)==0)
        rEst = phiapx.step(uEst);        
        vdv.step(rEst);
        vdpobs.step(vObs,msrProc((rEst),'Forward'));
        vdp.step(uEst,rEst);
        set(hTitle3,'String',sprintf('Rfl Est (%3d): MSE = %6.4e',itr,mymse(rEst,phi.step(uSrc))));
        drawnow
    end
    toc
end
lamda = pdshshc.LambdaCompensated;
eta   = pdshshc.EtaCompensated;
disp(['lambda = ' num2str(lambda)]);
disp(['eta    = ' num2str(eta)]);

%% 結果表示
rSrc = phi.step(uSrc);
fprintf('Restoration of Reflection: MSE = %6.4e\n',mymse(rEst,rSrc));

%%
dt = char(datetime('now','TimeZone','local','Format','d-MM-y-HH-mm-ssZ'));
targetdir = ['./sim_clg_' dt];
if exist(targetdir,'dir') ~= 7
    mkdir(targetdir)
end
save([targetdir '/uest_sim'],'uEst')
save([targetdir '/rest_sim'],'rEst')
save([targetdir '/usrc_sim'],'uSrc')
save([targetdir '/rsrc_sim'],'rSrc')
save([targetdir '/vobs_sim'],'vObs')

%% データ保存
options.method = method;
options.barlambda = barlambda;
options.bareta    = bareta;
options.lambda = lambda;
options.eta    = eta;
options.gamma1 = gamma1;
options.phiMode = phiMode;
options.vrange = vrange;
options.msrProc = msrProc;
options.fwdDic = fwdDic;
options.adjDic = adjDic;
options.gdnFcnG = gdnFcnG;
options.gdnFcnH = gdnFcnH;
save([targetdir '/results_sim' ],'options') % TODO XML?