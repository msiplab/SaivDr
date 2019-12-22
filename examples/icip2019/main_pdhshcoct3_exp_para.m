% 範囲制約付き OCT ボリュームデータ復元（実験）
%
% Reference:
%
% S. Ono, "Primal-dual plug-and-play image restoration,"
% IEEE Signal Processing Letters, vol.24, no.8, pp.1108-1112, Aug. 2017.
%isCoin = false;

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
splitfactor = [2 2 20]; %2*ones(1,3);  % 垂直・水平・奥行方向並列度
%splitfactor = [1 1 1]; %2*ones(1,3);  % 垂直・水平・奥行方向並列度
padsize = 2^(nLevels-1)*ones(1,3); % OLS/OLA パッドサイズ
isintegritytest = false; % 整合性テスト
useparallel = false; % 並列化
usegpu = true; % GPU
issingle = true; % 単精度
method = 'udht';
%method = 'bm4d';
isNoDcShk   = false;  % DC成分閾値回避
%isEnvWght   = false;  % 包絡線重み付け

%% 最大繰り返し回数
maxIter = 1000;

%% 可視化パラメータ
isVerbose  = true;
isVisible  = true;  % グラフ表示
monint     = 50;    % モニタリング間隔
texture    = '2D';
slicePlane = 'YZ';  % 画像スライス方向
daspect    = [1 1 10];
isIsta     = false; % ISTA
obsScale = 20; % 観測データ用モニタの輝度調整
estScale = 200; % 復元データ用モニタの輝度調整
vdpScale = [1 10]; % プロット用スケール調整

%%
dt = char(datetime('now','TimeZone','local','Format','d-MM-y-HH-mm-ssZ'));
targetdir = ['./exp' dt];
if exist(targetdir,'dir') ~= 7
    mkdir(targetdir)
end

%% 最適化パラメータ
% eta = 0, vmin=-Inf, vmax=Inf で ISTA相当
sizecomp = true; % 正則化パラメータのサイズ補正
barlambda = 4e-12; % 正則化パラメータ (変換係数のスパース性）
bareta    = 4e-9; % 正則化パラメータ（奥行方向の全変動）
vrange   = [1.00 1.50]; % ハード制約下限
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
%
pScale = 8.00; % 光強度
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
    adjDic = UdHaarAnalysis3dSystem('NumLevels',nLevels);
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

%% 観測データ
nSlices = 3000;
cSlice  = 1000;
%nSubSlices = 16;
nSubSlices = 2000;
sY = 1;
eY = 256;
sX = 1;
eX = 256;
sdir = '../LEXAR';
adjScale  = 5e-3;
%{
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
    %nSubSlices = 16;
    nSubSlices = 1600;
    sY = 1;
    eY = 244;
    sX = 1;
    eX = 240;
    sdir = '../0519_S0007生体村松研用';
    adjScale  = 1e2;
end
%}

%% データ読み込み
data = fopen(sprintf('%s/oct0_C001H001S0001-00.mraw',sdir));
b=fread(data,'uint16');
nDims = [(eY-sY+1) (eX-sX+1) nSlices];
array3d = zeros(nDims);

hw = waitbar(0,'now reading and decording');
for i=1:nDims(1)
   for j=1000:nDims(3)
       array3d(i,:,j-1000+1) = b(1+(i-1)*nDims(2)+(j-1)*nDims(2)*nDims(1):i*nDims(2)+(j-1)*nDims(2)*nDims(1));
   end
   waitbar(i/nDims(1),hw)
end
close(hw)
%array3d = array3d/max(abs(array3d(:)));
%{
finfo = imfinfo(sprintf('%s/k_C001H001S00010001.tif',sdir));
array3d = zeros(finfo.Height,finfo.Width,nSlices);
hw = waitbar(0,'Load data...');
for iSlice = 1:nSlices
    fname = sprintf('%s/k_C001H001S0001%04d.tif',sdir,iSlice);
    array3d(:,:,iSlice) = im2double(imread(fname));
    waitbar(iSlice/nSlices,hw);
end
close(hw)
%}

%% 奥行方向ハイパスフィルタ
nLen = 21;
lpf = ones(nLen,1)/nLen;
lpf = permute(lpf,[2 3 1]);
%array3d = array3d - imfilter(array3d,lpf,'symmetric');

%% 処理対象の切り出し
nDim = size(array3d);
%vObs = array3d(sY:eY,sX:eX,cSlice-nSubSlices/2+1:cSlice+nSubSlices/2); % TODO: SUBVOLUME
limits = [sX eX sY eY (cSlice-nSubSlices/2+1) (cSlice+nSubSlices/2)];
array3d = subvolume(array3d,limits);
vObs = array3d - imfilter(array3d,lpf,'symmetric');
vObs = adjScale*vObs;
[height, width, depth] = size(vObs);
clear array3d
if issingle
    vObs = single(vObs);
end

vobsname = [targetdir '/vobs'];
save(vobsname,'vObs');

%% 観測過程生成
msrProc = Coherence3(...
    'Scale',pScale,...
    'Sigma',pSigma,...
    'Frequency',pFreq,...
    'UseGpu',usegpu);

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
        'Texture',texture,...
        'VRange',[-1 1],...
        'Scale',obsScale);    
    if strcmp(texture,'2D')
        vdvobs.SlicePlane = slicePlane;
        title(sprintf('Obs %s slice',slicePlane))
    else
        vdvobs.DAspect = daspect;
        title('Obs volume')
    end 
    vdvobs.step(vObs);

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
    subplot(2,2,2)
    vdv.step(r);
    hTitle2 = title('Rfl Est(  0)');
    %
    subplot(2,2,4)
    vdp = VolumetricDataPlot('Direction','Z','Scales',vdpScale,'NumPlots',2);
    vdp.step(vObs,r);
    axis([0 size(vObs,3) -1 2])
    legend('Refraction Idx',['Reflectance x' num2str(vdpScale(2))],'Location','best')
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
    'GaussianDenoiser', { gdnFcnG, gdnFcnH } ,...
    'SplitFactor',    splitfactor,...  % 垂直・水平・奥行方向並列度
    'PadSize',        padsize,...
    'UseParallel',    useparallel,...
    'UseGpu',         usegpu,...
    'IsIntegrityTest', isintegritytest);

disp(pdshshc)

%% 復元処理
id = 'LEXAR';
%if isCoin
%    id = '0519_S0002';
%else
%    id = '0519_S0007';
%end
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
        set(hTitle2,'String',sprintf('Rfl Est (%3d)',itr));
        save([targetdir '/uest_exp_' id '_' num2str(itr)],'uEst')
        save([targetdir '/rest_exp_' id '_' num2str(itr)],'rEst') 
        drawnow
    end
    toc
end
lambda = pdshshc.LambdaCompensated;
eta   = pdshshc.EtaCompensated;
disp(['lambda = ' num2str(lambda)]);
disp(['eta    = ' num2str(eta)]);

%% 推定データ保存

save([targetdir '/uest_exp_' id ],'uEst')
save([targetdir '/rest_exp_' id ],'rEst')

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
save([targetdir '/results_exp_' id],'options')