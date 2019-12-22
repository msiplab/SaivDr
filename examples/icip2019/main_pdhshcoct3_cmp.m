% 範囲制約付き OCT ボリュームデータ復元（比較検討用シミュレーション）
%
% Reference:
%
% S. Ono, "Primal-dual plug-and-play image restoration,"
% IEEE Signal Processing Letters, vol.24, no.8, pp.1108-1112, Aug. 2017.
%
nTrials    = 5;

method = 'udht';
% method = 'bm4d';

% データ保存フォルダ
dt = char(datetime('now','TimeZone','local','Format','d-MM-y-HH-mm-ssZ'));
targetdir = ['./data' dt];
if exist(targetdir,'dir') ~= 7
    mkdir(targetdir)
end

% 可視化パラメータ
isVerbose  = false; % グラフ表示

%%
maxIter = 1e+3;        % 最大繰返し回数
vrange  = [1.00 1.50]; % ハード制約
gamma1  = 1e-3;        % ステップサイズ
% 
% PHIMODE in { 'Linear', 'Signed-Quadratic', 'Reflection' }
phiMode   = 'Linear';

%% 
if strcmp(method,'udht')
    dicSet = { 'udht' }; % 辞書
    gdnSet = { 'sfth' }; % ガウスデノイザ
    levSet = { 1, 2, 3 };
    % eta = 0, vmin=-Inf, vmax=Inf
elseif strcmp(method,'bm4d')
    dicSet = { 'idnt' }; % 辞書
    gdnSet = { 'bm4d' }; % ガウスデノイザ
    levSet = { 1 };
else
    error('METHOD')
end
% eta = 0, vmin=-Inf, vmax=Inf
lmdSet = num2cell(2.^(0:6)*1e-5);
etaSet = num2cell(2.^(0:6)*1e-1);
ndcSet = { false }; %
envSet = { false }; %
awgSet = { 0.04 };  % 観測ノイズ

% 比較パラメータ数
nDics = length(dicSet);
nGdns = length(gdnSet);
nLevs = length(levSet);
nAwgs = length(awgSet);
nLmds = length(lmdSet);
nEtas = length(etaSet);
nNdcs = length(ndcSet);
nEnvs = length(envSet);

nParamSet = nDics*nLevs*nLmds*nEtas*nNdcs*nEnvs*nAwgs;

%% 観測パラメータ設定
pScale = 8.00; % 光強度
pSigma = 8.00; % 広がり
pFreq  = 0.25; % 周波数

%% 原信号データ
depth   = 64;    % 奥行
height  = depth; % 高さ
width   = 16; % 幅
phtm    = phantom('Modified Shepp-Logan',depth);
sliceYZ = permute(phtm,[1 3 2]);
uSrc    = 0.5*repmat(sliceYZ,[1 width 1]) + 1;

%% 設定条件
save([targetdir '/usrc'],'uSrc')
phi  = RefractIdx2Reflect();
rSrc = phi.step(uSrc);
save([targetdir '/rsrc'],'rSrc')

msrProc = Coherence3(...
    'Scale',pScale,...
    'Sigma',pSigma,...
    'Frequency',pFreq);

%% 設定条件
paramSet   = cell(nParamSet,1);
volumeData = cell(nParamSet,1);

mymse = @(x,y) norm(x(:)-y(:),2)^2/numel(x);
mymse_ = cell(nParamSet,nTrials);
for iTrial = 1:nTrials
    
    idx = 1;
    for iAwg = 1:nAwgs
        wSigma = awgSet{iAwg};
        % 設定条件
        vObs = msrProc.step(phi.step(uSrc),'Forward') ...
            + wSigma*randn(size(uSrc));
        vobsname = [targetdir strrep(sprintf('/vobs_ref_wsigma%0.3e_trial%03d',...
            wSigma,iTrial),'.','_')];
        save(vobsname,'vObs');
        for iLmd = 1:nLmds
            lambda = lmdSet{iLmd};
            for iEta = 1:nEtas
                eta = etaSet{iEta};
                for iNdc = 1:nNdcs
                    isNoDcShk = ndcSet{iNdc};
                    for iEnv = 1:nEnvs
                        isEnvWght = envSet{iEnv};
                        for iGdn = 1:nGdns
                            gdn = gdnSet{iGdn};
                            for iDic = 1:nDics
                                dic = dicSet{iDic};
                                for iLev = 1:nLevs
                                    nLevels = levSet{iLev};
                                    if ~(strcmp(dic,'idnt') && nLevels > 1)
                                        paramSet{idx}.nLevels   = nLevels;
                                        paramSet{idx}.wSigma    = wSigma;
                                        paramSet{idx}.maxIter   = maxIter;
                                        paramSet{idx}.vrange    = vrange;
                                        paramSet{idx}.phiMode   = phiMode;
                                        paramSet{idx}.gamma1    = gamma1;
                                        paramSet{idx}.lambda    = lambda;
                                        paramSet{idx}.eta       = eta;
                                        paramSet{idx}.isNoDcShk = isNoDcShk;
                                        paramSet{idx}.isEnvWght = isEnvWght;
                                        paramSet{idx}.gdn       = gdn;
                                        paramSet{idx}.dic       = dic;
                                        volumeData{idx}.vObs    = vObs;
                                        idx = idx+1;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    nParamSet = idx-1;
    
    parsave = @(fname,x) save(fname,'x');
    parfor idx = 1:nParamSet
        lambda    = paramSet{idx}.lambda;
        eta       = paramSet{idx}.eta;
        isNoDcShk = paramSet{idx}.isNoDcShk;
        isEnvWght = paramSet{idx}.isEnvWght;
        nLevels   = paramSet{idx}.nLevels;
        gdn       = paramSet{idx}.gdn;
        dic       = paramSet{idx}.dic;
        %
        vObs      = volumeData{idx}.vObs;
        
        % Dictionary
        if strcmp(dic,'udht')
            import saivdr.dictionary.udhaar.*
            fwdDic = UdHaarSynthesis3dSystem();
            adjDic = UdHaarAnalysis3dSystem('NumberOfLevels',nLevels);
        elseif strcmp(dic,'idnt')
            import saivdr.dictionary.utility.*
            fwdDic = IdentitySynthesisSystem();
            adjDic = IdentityAnalysisSystem('IsVectorize',false);
        else
            error('DIC')
        end
        
        % Gaussian Denoiser
        if strcmp(gdn,'sfth')
            import saivdr.restoration.denoiser.*
            gdnFcnG = GaussianDenoiserSfth();
            gdnFcnH = GaussianDenoiserSfth();
        elseif strcmp(gdn,'bm4d')
            import saivdr.restoration.denoiser.*
            gdnFcnG = GaussianDenoiserBm4d();
            gdnFcnH = GaussianDenoiserSfth();
        else
            error('GDN')
        end
        
        % 設定条件
        fname = strrep(...
            sprintf('dic%s_lev%d_gdn%s_lmd%0.3e_eta%0.3e_nds%d_env%d_trial%03d',...
            dic,nLevels,gdn,lambda,eta,isNoDcShk,isEnvWght,iTrial),...
            '.','_');
        disp(fname)
        
        % 復元ステップ
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
        % 屈折率反射率変換
        phiapx = RefractIdx2Reflect(...
            'PhiMode',phiMode,...
            'VRange', vrange);
        
        % 復元処理
        for itr = 1:maxIter
            uEst = pdshshc.step();
        end
        rEst = phiapx.step(uEst);
        
        % 推定データ保存
        parsave([targetdir '/uest_' fname ],uEst)
        parsave([targetdir '/rest_' fname ],rEst)
        
        % 結果保存
        mymse_{idx,iTrial} = mymse(rEst,rSrc);
    end
    
    %% データ保存
    save([targetdir '/results_cmp'],'mymse_','paramSet')
end