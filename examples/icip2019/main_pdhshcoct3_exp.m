% 範囲制約付き OCT ボリュームデータ復元（実験）
%
% Reference:
%
% S. Ono, "Primal-dual plug-and-play image restoration,"
% IEEE Signal Processing Letters, vol.24, no.8, pp.1108-1112, Aug. 2017.
isCoin = false;

isVisible   = true;  % グラフ表示
slicePlane  = 'YZ';  % 画像スライス方向
isIsta      = false; % ISTA
isNoDcShk   = false;  % DC成分閾値回避
%isEnvWght   = false;  % 包絡線重み付け
isVerbose   = true;

maxIter = 1000;
nLevels = 1;

obsScale = 20;
estScale = 40;

% method
method = 'udht';
%method = 'bm4d';

%%
dt = char(datetime('now','TimeZone','local','Format','d-MM-y-HH-mm-ssZ'));
targetdir = ['./exp' dt];
if exist(targetdir,'dir') ~= 7
    mkdir(targetdir)
end

%% 最適化パラメータ
% eta = 0, vmin=-Inf, vmax=Inf で ISTA相当
lambda  = 1e-5; % 正則化パラメータ (変換係数のスパース性）
eta     = 1e-1; % 正則化パラメータ（奥行方向の全変動）
vrange  = [1.00 1.50]; % ハード制約下限
%
gamma1  = 1e-3; % ステップサイズ TODO 条件を確認
%options.maxIter = 1e+3; % 最大繰返し回数
%options.stopcri = 1e-9; % 停止条件
% 観測モデル（復元処理のみ）
% PHIMODE in { 'Linear', 'Signed-Quadratic', 'Reflection' }
phiMode   = 'Linear'; 
%phiMode   = 'Signed-Quadratic'; % γ1を小さく 1e-7
%phiMode   = 'Reflection'; 　　　% γ1を小さく 1e-30
%
% TODO 非線形モデルは勾配消失問題にあたるよう。γ1の制御が必要か。
%

%% 観測パラメータ設定
% ICIP2018
pScale = 1.00; % 光強度
pSigma = 8.00; % 広がり
pFreq  = 0.25; % 周波数
% SIPシンポ2017
%{
pScale = 1.00;
pSigma = 2.00;
pFrq   = 0.20;
%}
% ICASSP2018
%{
pScale = 1.00;
pSigma = 8.00;
pFrq   = 0.25;
%}

%% 合成辞書（パーセバルタイトフレーム）＆ ガウスノイズ除去
if strcmp(method,'udht')
    import saivdr.dictionary.udhaar.*
    import saivdr.restoration.denoiser.*
    fwdDic = UdHaarSynthesis3dSystem();
    adjDic = UdHaarAnalysis3dSystem('NumberOfLevels',nLevels);
    gdnFcnG = GaussianDenoiserSfth();
    gdnFcnH = GaussianDenoiserSfth();
elseif strcmp(method,'bm4d')
    import saivdr.dictionary.udhaar.*
    import saivdr.restoration.denoiser.*
    fwdDic = IdentitySynthesisSystem();
    adjDic = IdentityAnalysisSystem('IsVectorize',false);
    gdnFcnG = GaussianDenoiserBm4d();
    gdnFcnH = GaussianDenoiserSfth();
else
    error('METHOD')
end

%% 観測データ
if isCoin
    nSlices = 1211;
    cSlice  = 500;
    nSubSlices = 8;
    %nSubSlices = 512;
    sdir = '../0519_S0002coin村松研用';
    sY = 201;
    eY = 440;
    sX = 181;
    eX = 420;
    adjScale  = 5e1;
else
    nSlices = 1673;
    cSlice  = 836;
    nSubSlices = 256;
    %nSubSlices = 1024;
    sY = 1;
    eY = 244;
    sX = 1;
    eX = 240;
    sdir = '../0519_S0007生体村松研用';
    adjScale  = 1e2;
end

%% データ読み込み
finfo = imfinfo(sprintf('%s/k_C001H001S00010001.tif',sdir));
array3d = zeros(finfo.Height,finfo.Width,nSlices);
hw = waitbar(0,'Load data...');
for iSlice = 1:nSlices
    fname = sprintf('%s/k_C001H001S0001%04d.tif',sdir,iSlice);
    array3d(:,:,iSlice) = im2double(imread(fname));
    waitbar(iSlice/nSlices,hw);
end
close(hw)

%% 奥行方向ハイパスフィルタ
nLen = 21;
lpf = ones(nLen,1)/nLen;
lpf = permute(lpf,[2 3 1]);
array3d = array3d - imfilter(array3d,lpf,'circular');

%% 処理対象の切り出し
nDim = size(array3d);
%vObs = array3d(sY:eY,sX:eX,cSlice-nSubSlices/2+1:cSlice+nSubSlices/2); % TODO: SUBVOLUME
limits = [sX eX sY eY (cSlice-nSubSlices/2+1) (cSlice+nSubSlices/2)];
vObs = subvolume(array3d,limits);
vObs = adjScale*vObs;
[height, width, depth] = size(vObs);
clear array3d

vobsname = [targetdir '/vobs'];
save(vobsname,'vObs');

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

%% 観測データ表示
if isVisible
    import saivdr.utility.*
    % 準備
    hImg = figure(1);
    subplot(2,2,1)
    vdvobs = VolumetricDataVisualizer(...
        'Texture','3D',...
        'DAspect',[1 1 3],...
        'VRange',[-1 1],...
        'Scale',obsScale);    
    vdvobs.step(vObs);
    title(sprintf('Obs %s slice',slicePlane))
    %
    subplot(2,2,3)
    vdpobs = VolumetricDataPlot('Direction','Z','NumPlots',2);
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
end

%% Z方向包絡線検出 → η設定
%{
if isEnvWght
    % フィルタリング
    v = fwdProc(adjProc(vObs));

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
        'Texture','3D',...
        'DAspect',[1 1 3],...
        'VRange',[-1 1],...
        'Scale',estScale);    
    phiapx = RefractIdx2Reflect(...
        'PhiMode',phiMode,...
        'VRange', vrange);
    r = phiapx.step(vObs);
    %
    figure(hImg)
    subplot(2,2,2)
    vdv.step(r);
    hTitle2 = title('Rfl Est(  0)');
    %
    subplot(2,2,4)
    vdp = VolumetricDataPlot('Direction','Z','Scales',[1 10],'NumPlots',2);
    vdp.step(vObs,r);
    axis([0 size(vObs,3) -1 2])
    legend('Refraction Idx','Reflectance x10','Location','best')
    title('Restoration')
end

%% 復元システム生成
pdshshc = PdsHsHcOct3(...
    'Observation',    vObs,...
    'Lambda',         lambda,...
    'Eta',            eta,...
    'Gamma1',         gamma1,...
    'PhiMode',        phiMode,...
    'VRange',         vrange,...
    'MeasureProcess', msrProc,...
    'Dictionary',     { fwdDic, adjDic },...
    'GaussianDenoiser', { gdnFcnG, gdnFcnH } );

disp(pdshshc)

%% 復元処理
for itr = 1:maxIter
    uEst = pdshshc.step();
    % Monitoring
    vdv.step(phiapx.step(uEst));
    vdpobs.step(vObs,msrProc(phiapx.step(uEst),'Forward'));
    vdp.step(uEst,phiapx.step(uEst));
    set(hTitle2,'String',sprintf('Rfl Est (%3d)',itr));
    %
    drawnow    
end

%% 結果表示
rEst = phiapx.step(uEst);
fprintf('Restoration of Reflection\n');

%% 推定データ保存
if isCoin
    id = '0519_S0002';
else
    id = '0519_S0007';
end
save([targetdir '/uest_exp_' id ],'uEst')
save([targetdir '/rest_exp_' id ],'rEst')

%% データ保存
%save([targetdir '/results_exp_' id],'options')
