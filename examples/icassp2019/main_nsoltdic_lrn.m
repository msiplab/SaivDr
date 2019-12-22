% MAIN_NSOLTDIC_LRN
% NSOLT辞書学習スクリプト
%
% SaivDr パッケージにパスを通してください。
%
% 例
%
% >> setpath 
%
% 測定結果(VTK)を　
%
%   '../DATA(20170906)/VTK/';
%
% に用意して、以下のコマンドでポイントクラウド化してください。
%
% >> main_vtk2pcd
%

%% コンフィグレーション
isVisible = true;
diclrnconfig.srcFolder = RiverCpsConfig.SrcFolder;
diclrnconfig.dicFolder = RiverCpsConfig.DicFolder;

%% 時間設定
diclrnconfig.ts = RiverCpsConfig.TsTraining; % start
diclrnconfig.te = RiverCpsConfig.TeTraining; % end
diclrnconfig.ti = RiverCpsConfig.TiTraining; % interval

%% 可視化パラメータ
scale = 1024;
map = colormap();

%% 学習用データフィールド
diclrnconfig.fieldList = RiverCpsConfig.FieldListTraining;
nFields = numel(diclrnconfig.fieldList);

%% データ読み込みと整形
iFrame = 0;
nFrames = (diclrnconfig.te-diclrnconfig.ts)/diclrnconfig.ti + 1;
frameSeq = cell(nFrames,1);
for t = diclrnconfig.ts:diclrnconfig.ti:diclrnconfig.te
    iFrame = iFrame + 1;
    filename = sprintf('%04d_trm',t);
    disp(filename)
    for iField = 1:nFields
        field = diclrnconfig.fieldList{iField};
        ptCloud = pcread([ diclrnconfig.srcFolder field '_' filename '.pcd' ]);
        undImg = permute(ptCloud.Location(:,:,3),RiverCpsConfig.getDimOrd());
        frameSeq{iFrame}(:,:,iField) = undImg;
    end
end

%% 訓練データ表示
if isVisible
    hImg = cell(nFields,1);
    hFig1 = figure(1);
    for iFrame = 1:nFrames
        undImg = frameSeq{iFrame};
        if iFrame == 1
            figure(hFig1)
            for iField = 1:nFields
                subplot(nFields,1,iField)
                hImg{iField} = imshow(fliplr(scale*undImg(:,:,iField)),map);
            end
        else
            for iField = 1:nFields
                hImg{iField}.CData = fliplr(scale*undImg(:,:,iField));
            end
        end
        pause(0.1)
        drawnow
    end
end

%% NSOLT訓練データ準備
% 深さは2で固定
diclrnconfig.nSubPtcs  = RiverCpsConfig.NumPatchesTraining; % # of patches
diclrnconfig.virWidth  = RiverCpsConfig.VirWidthTraining;   % Virtual width of training images
diclrnconfig.virLength = RiverCpsConfig.VirLengthTraining;  % Virtual height of training images

% 訓練データのランダム抽出
diclrnconfig.width  = size(frameSeq{1},RiverCpsConfig.DIRECTION_WIDTH);
diclrnconfig.length = size(frameSeq{1},RiverCpsConfig.DIRECTION_LENGTH);
diclrnconfig.depth  = 2;
%
trnImgs = cell(diclrnconfig.nSubPtcs);
rng(0,'twister');
for iSubPtc = 1:diclrnconfig.nSubPtcs
    pl = randi([0 (diclrnconfig.length-diclrnconfig.virLength)]); % 流下方向
    pw = randi([0 (diclrnconfig.width-diclrnconfig.virWidth)]);   % 横断方向
    pf = randi([1 nFrames]);            % フレーム
    %
    undImg = frameSeq{pf};
    trnImgs{iSubPtc} = ...
        im2double(undImg(pw+(1:diclrnconfig.virWidth),pl+(1:diclrnconfig.virLength),:));
end

%% 
if RiverCpsConfig.IsVisible
    hfig1 = findobj(get(groot,'Children'),'Name','Sparse Approximation');
    if isempty(hfig1)
        hfig1 = figure;
        set(hfig1,'Name','Sparse Approximation')
    end
    %
    hfig2 = findobj(get(groot,'Children'),'Name','Atomic Images');
    if isempty(hfig2)
        hfig2 = figure;
        hfig2.Name= 'Atomic Images';
    end
end

%% NSOLT学習(ISTA+SGD)
diclrnconfig.nIters = RiverCpsConfig.NumberOfOuterIterations; % 繰り返し回数
plotFcn             = @optimplotfval;

% ISTAステップシステムのインスタンス化
import saivdr.restoration.ista.IstaSystem
algorithm = IstaSystem(...
    'DataType','Volumetric Data',...
    'Lambda',RiverCpsConfig.LambdaNsoltTraining);

% ステップモニターシステムのインスタンス化
import saivdr.utility.StepMonitoringSystem
stepMonitor = StepMonitoringSystem(...
    'DataType','Volumetric Data',...    
    'EvaluationType','double',...
    'ImageFigureHandle',hfig1,...
    'IsRMSE',true,...
    'IsMSE',true,...    
    'IsVisible',false,...RiverCpsConfig.IsVisible,...
    'IsVerbose',RiverCpsConfig.IsVerbose);

% スパース近似システムのインスタンス化
import saivdr.sparserep.IterativeSparseApproximater
sprsAprx = IterativeSparseApproximater(...
    'Algorithm',algorithm,...
    'StepMonitor',stepMonitor,...
    'MaxIter',RiverCpsConfig.MaxIterOfIterativeSparseApproximater,...
    'TolErr',RiverCpsConfig.TolErr);

% 辞書更新システムのインスタンス化
import saivdr.dictionary.nsoltx.design.NsoltDictionaryUpdateSgd
dicUpd = NsoltDictionaryUpdateSgd(...
    'IsVerbose', RiverCpsConfig.IsVerbose,...
    'GradObj',RiverCpsConfig.GradObj,...
    'Step',RiverCpsConfig.SgdStep,....
    'AdaGradEta',RiverCpsConfig.AdaGradEta,...
    'AdaGradEps',RiverCpsConfig.AdaGradEps);

% 辞書学習システムのインスタンス化
import saivdr.dictionary.nsoltx.design.NsoltDictionaryLearningPnP
designer = NsoltDictionaryLearningPnP(...
    'DataType', 'Volumetric Data',...
    'DecimationFactor',RiverCpsConfig.DecimationFactor,...
    'NumberOfLevels',RiverCpsConfig.NumberOfLevels,...    
    'NumberOfChannels', RiverCpsConfig.NumberOfChannels,...
    'PolyPhaseOrder',RiverCpsConfig.PolyPhaseOrder,...
    'NumberOfVanishingMoments',RiverCpsConfig.NumberOfVanishingMoments,...
    'SparseApproximater',sprsAprx,...
    'DictionaryUpdater',dicUpd,...
    'IsRandomInit', RiverCpsConfig.IsRandomInit,...
    'StdOfAngRandomInit', RiverCpsConfig.StdOfAngRandomInit);
%
diclrnconfig.options = optimset(...
    'Display','iter',...
    'PlotFcn',plotFcn,...
    'MaxIter',RiverCpsConfig.MaxIterOfDictionaryUpdater);

%% 辞書学習
for iter = 1:diclrnconfig.nIters
    [ nsolt, mse ] = designer.step(trnImgs,diclrnconfig.options);
    fprintf('MSE (%d) = %g\n',iter,mse)
    if isVisible
        % Show the atomic images by using a method atmimshow()
        figure(hfig2)
        nsolt.atmimshow()
        drawnow
    end
end

%%
imgName = 'rivercps';
if ~exist(diclrnconfig.dicFolder,'dir')
    mkdir(diclrnconfig.dicFolder);
end
fileName = sprintf(...
    '%snsolt_d%dx%dx%d_c%d+%d_o%d+%d+%d_v%d_lv%d_lmd%s_%s_sgd.mat',...
    diclrnconfig.dicFolder,...
    RiverCpsConfig.DecimationFactor(1),...
    RiverCpsConfig.DecimationFactor(2),...
    RiverCpsConfig.DecimationFactor(3),...
    RiverCpsConfig.NumberOfChannels(1),...
    RiverCpsConfig.NumberOfChannels(2),...
    RiverCpsConfig.PolyPhaseOrder(1),...
    RiverCpsConfig.PolyPhaseOrder(2),...
    RiverCpsConfig.PolyPhaseOrder(3),...
    RiverCpsConfig.NumberOfVanishingMoments,...
    RiverCpsConfig.NumberOfLevels,...
    strrep(num2str(RiverCpsConfig.LambdaNsoltTraining,'%g'),'.','_'),...
    [imgName num2str(diclrnconfig.virLength) 'x' num2str(diclrnconfig.virWidth)]);
save(fileName,'nsolt','designer','mse','diclrnconfig');