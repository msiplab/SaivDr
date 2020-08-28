classdef TestCasePdPnPOct3 < matlab.unittest.TestCase
    %TESTCASEPDPNPOCT3 このクラスの概要をここに記述
    %   詳細説明をここに記述
    
    properties (TestParameter)
        %         scl  = struct('small',0.1, 'large', 10.0);
        %         sgm  = struct('small',0.1, 'large', 10.0);
        useparallel = struct('true', true, 'false', false );        
        niter   = struct('small',1, 'large', 4 );
        vrange  = struct('low',[1.0 1.5], 'high', [1.5 2.0]);
        depth = struct('small',8, 'large', 32);
        width = struct('small',8, 'large', 32);
        dsplit = struct('small',1, 'large', 4);
        nlevels = { 1, 3 };
    end
    
    methods (Test)
        
        function testConstruction(testCase)
            
            % パラメータ（期待値）
            lambdaExpctd  = 0.01;     % 正則化パラメータ
            gamma1Expctd  = 0.01;     % ステップサイズ
            vrangeExpctd  = [ -1.00 1.00 ];     % ハード制約下限
            isNoDcShrinkExpctd = false;  % 直流ソフト閾値処理回避
            
            % インスタンス生成
            target = PdPnPOct3();
            
            % 実現値
            lambdaActual  = target.Lambda;  % 正則化パラメータ
            gamma1Actual  = target.Gamma1;  % ステップサイズ
            vrangeActual  = target.VRange;  % ハード制約下限
            isNoDcShrinkActual = target.IsNoDcShrink;  % 直流ソフト閾値処理回避
            
            % 評価
            testCase.verifyEqual(lambdaActual,lambdaExpctd);
            testCase.verifyEqual(gamma1Actual,gamma1Expctd);
            testCase.verifyEqual(vrangeActual,vrangeExpctd);
            testCase.verifyEqual(isNoDcShrinkActual,isNoDcShrinkExpctd);
            
        end
        
        function testStep(testCase,...
                depth,width,nlevels,niter)
            
            % パラメータ
            phtm = phantom('Modified Shepp-Logan',depth);
            sliceYZ = permute(phtm,[1 3 2]);
            uSrc = 0.5*repmat(sliceYZ,[1 width 1]) + 1;
            
            % 期待値
            iterExpctd = niter;
            
            % 観測データ生成
            wSigma = 4e-2; % ノイズ分散
            pScale = 8.00; % 光強度
            pSigma = 8.00; % 広がり
            pFreq  = 0.25; % 周波数
            coh3 = Coherence3(...
                'Scale',pScale,...
                'Sigma',pSigma,...
                'Frequency',pFreq);
            phi  = RefractIdx2Reflect();
            vObs = coh3.step(phi.step(uSrc),'Forward') ...
                + wSigma*randn(size(uSrc));
            
            % インスタンス生成
            import saivdr.dictionary.udhaar.*
            import saivdr.restoration.denoiser.*
            fwdDic  = UdHaarSynthesis3dSystem();
            adjDic  = UdHaarAnalysis3dSystem('NumberOfLevels',nlevels);
            gdnFcn  = GaussianDenoiserSfth();
            target  = PdPnPOct3(...
                'Observation',    vObs,...
                'MeasureProcess', coh3,...
                'Dictionary', { fwdDic, adjDic },...
                'GaussianDenoiser', gdnFcn );
            
            % 復元処理
            for iter = 1:iterExpctd
                r = target.step();
            end
            iterActual = target.Iteration;
            
            % 評価
            %import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(r,size(uSrc));
            testCase.verifyEqual(iterActual,iterExpctd)
            
        end
        
        function testStepBm4d(testCase,depth,width,niter)
            
            % パラメータ
            phtm = phantom('Modified Shepp-Logan',depth);
            sliceYZ = permute(phtm,[1 3 2]);
            uSrc = 0.5*repmat(sliceYZ,[1 width 1]) + 1;
            
            % 期待値
            iterExpctd = niter;
            
            % 観測データ生成
            wSigma = 4e-2; % ノイズ分散
            pScale = 8.00; % 光強度
            pSigma = 8.00; % 広がり
            pFreq  = 0.25; % 周波数
            coh3 = Coherence3(...
                'Scale',pScale,...
                'Sigma',pSigma,...
                'Frequency',pFreq);
            phi  = RefractIdx2Reflect();
            vObs = coh3.step(phi.step(uSrc),'Forward') ...
                + wSigma*randn(size(uSrc));
            
            % インスタンス生成
            import saivdr.dictionary.utility.*
            import saivdr.restoration.denoiser.*
            fwdDic  = IdentitySynthesisSystem();
            adjDic  = IdentityAnalysisSystem('IsVectorize',false);
            gdnFcn = GaussianDenoiserBm4d();
            target  = PdPnPOct3(...
                'Observation',    vObs,...
                'MeasureProcess', coh3,...
                'Dictionary', { fwdDic, adjDic },...
                'GaussianDenoiser', gdnFcn );
            
            % 復元処理
            for iter = 1:iterExpctd
                r = target.step();
            end
            iterActual = target.Iteration;
            
            % 評価
            %import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(r,size(uSrc));
            testCase.verifyEqual(iterActual,iterExpctd)
            
        end
        
        function testStepHdth(testCase,depth,width,niter,nlevels)
            
            % パラメータ
            phtm = phantom('Modified Shepp-Logan',depth);
            sliceYZ = permute(phtm,[1 3 2]);
            uSrc = 0.5*repmat(sliceYZ,[1 width 1]) + 1;
            
            % 期待値
            iterExpctd = niter;
            
            % 観測データ生成
            wSigma = 4e-2; % ノイズ分散
            pScale = 8.00; % 光強度
            pSigma = 8.00; % 広がり
            pFreq  = 0.25; % 周波数
            coh3 = Coherence3(...
                'Scale',pScale,...
                'Sigma',pSigma,...
                'Frequency',pFreq);
            phi  = RefractIdx2Reflect();
            vObs = coh3.step(phi.step(uSrc),'Forward') ...
                + wSigma*randn(size(uSrc));
            
            % インスタンス生成
            import saivdr.dictionary.udhaar.*
            import saivdr.restoration.denoiser.*
            fwdDic  = UdHaarSynthesis3dSystem();
            adjDic  = UdHaarAnalysis3dSystem('NumberOfLevels',nlevels);
            gdnFcn  = GaussianDenoiserHdth();
            target  = PdPnPOct3(...
                'Observation',    vObs,...
                'MeasureProcess', coh3,...
                'Dictionary', { fwdDic, adjDic },...
                'GaussianDenoiser', gdnFcn );
            
            % 復元処理
            for iter = 1:iterExpctd
                r = target.step();
            end
            iterActual = target.Iteration;
            
            % 評価
            %import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(r,size(uSrc));
            testCase.verifyEqual(iterActual,iterExpctd)
            
        end
        
        
        function testStepSplit(testCase,...
                depth,width,dsplit,nlevels,niter,useparallel)
            
            % パラメータ
            splitfactor = [2*ones(1,2) dsplit];
            padsize = 2^(nlevels-1)*ones(1,3);
            phtm = phantom('Modified Shepp-Logan',depth);
            sliceYZ = permute(phtm,[1 3 2]);
            uSrc = 0.5*repmat(sliceYZ,[1 width 1]) + 1;
            
            % 観測データ生成
            wSigma = 4e-2; % ノイズ分散
            pScale = 8.00; % 光強度
            pSigma = 8.00; % 広がり
            pFreq  = 0.25; % 周波数
            coh3 = Coherence3(...
                'Scale',pScale,...
                'Sigma',pSigma,...
                'Frequency',pFreq);
            phi  = RefractIdx2Reflect();
            vObs = coh3.step(phi.step(uSrc),'Forward') ...
                + wSigma*randn(size(uSrc));
            
            % インスタンス生成
            import saivdr.dictionary.udhaar.*
            import saivdr.restoration.denoiser.*
            fwdDic  = UdHaarSynthesis3dSystem();
            adjDic  = UdHaarAnalysis3dSystem('NumberOfLevels',nlevels);
            gdnFcn  = GaussianDenoiserSfth();
            
            reference  = PdPnPOct3(...
                'Observation',    vObs,...
                'MeasureProcess', coh3,...
                'Dictionary', { fwdDic, adjDic },...
                'GaussianDenoiser', gdnFcn);
                        
            target  = PdPnPOct3(...
                'Observation',    vObs,...
                'MeasureProcess', coh3,...
                'Dictionary', { fwdDic, adjDic },...
                'GaussianDenoiser', gdnFcn,...
                'SplitFactor',splitfactor,...
                'PadSize',padsize,...
                'UseParallel',useparallel);
            
            % 復元処理
            for iter = 1:niter
                resExpctd = reference.step();
                resActual = target.step();
            end
            
            % 評価
            %import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(resActual,size(resExpctd));
            diff = max(abs(resExpctd(:) - resActual(:)));
            testCase.verifyEqual(resActual,resExpctd,...
                'AbsTol',1e-4,sprintf('%g',diff));
            
        end
        
    end
    
end