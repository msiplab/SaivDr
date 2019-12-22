% MAIN_CSCDMD_RST
% 畳込みスパース符号化動的モード分解(CSC-DMD)復元スクリプト
%
% SaivDr パッケージにパスを通してください。
%
% 例
%
% >> setpath 
%
% NSOLTの設計とCSCDMDの学習を先に完了させてください。
%
% >> main_nsoltdic_lrn
% >> main_cscdmd_lrn
%

%% コンフィグレーション
isVisibule = true;
dmdrstconfig.srcFolder = RiverCpsConfig.SrcFolder;
dmdrstconfig.dstFolder = RiverCpsConfig.DstFolder;
dmdrstconfig.dicFolder = RiverCpsConfig.DicFolder;
dmdrstconfig.virLength = RiverCpsConfig.VirLengthTraining;
dmdrstconfig.virWidth = RiverCpsConfig.VirWidthTraining;

%% 時間設定
dmdrstconfig.ts = RiverCpsConfig.TsRestoration; % start
dmdrstconfig.te = RiverCpsConfig.TeRestoration; % end
dmdrstconfig.ti = RiverCpsConfig.TiRestoration; % interval

%% 設計データの読み込み
imgName = 'rivercps';
fileName = sprintf(...
    '%snsolt_d%dx%dx%d_c%d+%d_o%d+%d+%d_v%d_vl%d_vn%d_%s_sgd.mat',...
    dmdrstconfig.dicFolder,...
    RiverCpsConfig.DecimationFactor(1),...
    RiverCpsConfig.DecimationFactor(2),...
    RiverCpsConfig.DecimationFactor(3),...
    RiverCpsConfig.NumberOfChannels(1),...
    RiverCpsConfig.NumberOfChannels(2),...
    RiverCpsConfig.NumberOfPolyphaseOrder(1),...
    RiverCpsConfig.NumberOfPolyphaseOrder(2),...
    RiverCpsConfig.NumberOfPolyphaseOrder(3),...
    RiverCpsConfig.OrderOfVanishingMoment,...
    RiverCpsConfig.NumberOfLevels,...
    RiverCpsConfig.NumberOfSparseCoefsTraining,...
    [imgName num2str(dmdrstconfig.virLength) 'x' num2str(dmdrstconfig.virWidth)]);
S = load(fileName,'nsolt','Phi','b','lambda');
nsolt = S.nsolt;
Phi = S.Phi;
b = S.b;
lambda = S.lambda;

%% 分析・合成システムのインスタンス化
import saivdr.dictionary.nsoltx.NsoltFactory
analyzer    = NsoltFactory.createAnalysis3dSystem(nsolt);
synthesizer = NsoltFactory.createSynthesis3dSystem(nsolt);
analyzer.BoundaryOperation = 'Termination';
synthesizer.BoundaryOperation = 'Termination';

%% 復元用データフィールド
dmdrstconfig.fieldList = RiverCpsConfig.FieldListRestoration;
nFields = numel(dmdlrnconfig.fieldList);

%% 計測データの読み込みと整形
iFrame = 0;
nFrames = (dmdrstconfig.te-dmdrstconfig.ts)/dmdrstconfig.ti + 1;
surfObsvSeq = cell(nFrames,1);
bedExpctSeq = cell(nFrames,1);
nDec = RiverCpsConfig.DecimationFactor(1:2);
for t = dmdrstconfig.ts:dmdrstconfig.ti:dmdrstconfig.te
    iFrame = iFrame + 1;
    filename = sprintf('%04d_trm',t);
    disp(filename)
    % Surface
    field = dmdrstconfig.fieldList{1};
    ptCloud = pcread([ dmdrstconfig.srcFolder field '_' filename '.pcd' ]);
    undImg = permute(ptCloud.Location(:,:,3),RiverCpsConfig.getDimOrd());
    padSize = ceil(size(undImg)./nDec(:).').*nDec(:).'-size(undImg);
    surfObsvSeq{iFrame} = padarray(undImg,padSize,'post');    
    % Bed
    field = dmdrstconfig.fieldList{2};
    ptCloud = pcread([ dmdrstconfig.srcFolder field '_' filename '.pcd' ]);
    undImg = permute(ptCloud.Location(:,:,3),RiverCpsConfig.getDimOrd());
    padSize = ceil(size(undImg)./nDec(:).').*nDec(:).'-size(undImg);
    bedExpctSeq{iFrame} = padarray(undImg,padSize,'post');
end

%% スパース復元(IHT)
import saivdr.sparserep.IterativeHardThresholding
import saivdr.utility.StepMonitoringSystem
stepMonitor = StepMonitoringSystem(...
    'DataType','Volumetric Data',...
    'IsVerbose',true,...
    'IsMSE',true);
sparseCoder = IterativeHardThresholding(...
    'Synthesizer',synthesizer,...
    'AdjOfSynthesizer',analyzer,...
    'StepMonitor',stepMonitor);

%% スパース近似(IHT)の実行
dmdrstconfig.nSprsCoefs = RiverCpsConfig.NumberOfSparseCoefsEdmd;
for iFrame = 1:nFrames
    srcVol = frameSeq{iFrame};
    stepMonitor.SourceImage = srcVol;
    [~, sparseCoefs, setOfScales] = ...
        sparseCoder.step(srcVol,...
        dmdrstconfig.nSprsCoefs);
    featSeq(:,iFrame) = sparseCoefs(:);
    stepMonitor.release()
end