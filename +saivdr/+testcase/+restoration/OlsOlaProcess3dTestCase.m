classdef OlsOlaProcess3dTestCase < matlab.unittest.TestCase
    %OLSOLAPROCESS3DTESTCASE Test cases for OlsOlaProcess3d
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2018-2020, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    
    properties (TestParameter)
        useparallel = struct('true', true, 'false', false );
        usegpu = struct('true', true, 'false', false );        
        isintegrity = struct('true', true, 'false', false );
        width = struct('small', 32, 'large', 64);
        height = struct('small', 32, 'large', 64);
        depth = struct('small', 32,  'large', 64);
        vsplit = struct('small', 1, 'large', 4);
        hsplit = struct('small', 1,  'large', 4);
        dsplit = struct('small', 1, 'large', 4);
        level = struct('flat',1, 'sharrow',2,'deep', 3);
    end
    
    properties
        target
    end
    
    methods (TestMethodTeardown)
        function deteleObject(testCase)
            delete(testCase.target);
        end
    end
    
     
    methods (Static)
        function cpst = softthresh(x,cpre,lambda,gamma)
            u = cpre-gamma*x;
            v = abs(u)-lambda;
            cpst = sign(u).*(v+abs(v))/2;
        end
    end
    
    methods (Test)
        
        function testDefaultConstructor(testCase)
            
            % Expected values
            import saivdr.restoration.*
            analyzerExpctd = [];
            synthesizerExpctd = [];
            
            % Instantiation
            testCase.target = OlsOlaProcess3d();
            
            % Actual value
            analyzerActual = testCase.target.Analyzer;
            synthesizerActual = testCase.target.Synthesizer;
            
            % Evaluation
            testCase.assertEqual(analyzerActual,analyzerExpctd);
            testCase.assertEqual(synthesizerActual,synthesizerExpctd);
        end
        
        function testAnalyzerSynthesizer(testCase)
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            analyzerExpctd = UdHaarAnalysis3dSystem();
            synthesizerExpctd = UdHaarSynthesis3dSystem();
            
            % Instantiation
            import saivdr.restoration.*
            testCase.target = OlsOlaProcess3d(...
                'Analyzer',analyzerExpctd,...
                'Synthesizer',synthesizerExpctd);
            
            % Actual value
            analyzerActual = get(testCase.target,'Analyzer');
            synthesizerActual = get(testCase.target,'Synthesizer');
            
            % Evaluation
            testCase.assertEqual(analyzerActual,analyzerExpctd);
            testCase.assertEqual(synthesizerActual,synthesizerExpctd);
            
        end
        
        
        function testUdHaar(testCase,height,width,depth,level)
            
            % Parameters
            nLevels = level;
            srcImg = rand(height,width,depth);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer.NumberOfLevels = nLevels;
            
            % Instantiation of target class
            import saivdr.restoration.*
            testCase.target = OlsOlaProcess3d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer);
            
            % Actual values
            recImg = step(testCase.target,srcImg);
            
            % Evaluation
            testCase.verifySize(recImg,size(srcImg));
            diff = max(abs(srcImg(:) - recImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end

        % Test
        function testUdHaarSplittingSize(testCase,width,height,depth,...
                level,useparallel)
            
            % Parameters
            nLevels = level;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            nDepPad = 2^(level-1);
            srcImg = rand(height,width,depth);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer.NumberOfLevels = nLevels;
            
            % Instantiation of target class
            import saivdr.restoration.*
            testCase.target = OlsOlaProcess3d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...                
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Actual values
            recImg = step(testCase.target,srcImg);
            
            % Evaluation
            testCase.verifySize(recImg,size(srcImg));
            diff = max(abs(srcImg(:) - recImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-10,...
                sprintf('%g',diff));
        end
        
        % Test
        function testUdHaarSplittingSplit(testCase,vsplit,hsplit,dsplit,...
                level,useparallel)
            
            % Parameters
            nLevels = level;
            height_ = 48;
            width_ = 48;
            depth_ = 48;
            nVerSplit = vsplit;
            nHorSplit = hsplit;
            nDepSplit = dsplit;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            nDepPad = 2^(level-1);
            srcImg = rand(height_,width_,depth_);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer.NumberOfLevels = nLevels;
            
            % Instantiation of target class
            import saivdr.restoration.*
            testCase.target = OlsOlaProcess3d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Actual values
            recImg = step(testCase.target,srcImg);
            
            % Evaluation
            testCase.verifySize(recImg,size(srcImg));
            diff = max(abs(srcImg(:) - recImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-10,...
                sprintf('%g',diff));
        end
        
        % Test
        function testUdHaarSplittingWarningFactor(testCase,...
                width,height,depth,level,useparallel)
            
            % Parameters
            nLevels = level;
            nVerSplit = 3;
            nHorSplit = 3;
            nDepSplit = 3;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            nDepPad = 2^(level-1);
            srcImg = rand(height,width,depth);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer.NumberOfLevels = nLevels;
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalSplitFactorException';
            messageExpctd = 'Split factor must be a divisor of array size.';
            
            % Instantiation of target class
            import saivdr.restoration.*
            testCase.target = OlsOlaProcess3d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...              
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Evaluation
            try
                step(testCase.target,srcImg);
                if mod(width,nHorSplit) ~=0 || ...
                        mod(height,nVerSplit) ~= 0 || ...
                        mod(depth,nDepSplit) ~= 0
                    testCase.verifyFail(sprintf('%s must be thrown.',...
                        exceptionIdExpctd));
                end
            catch me
                switch me.identifier
                    case exceptionIdExpctd
                        messageActual = me.message;
                        testCase.verifyEqual(messageActual, messageExpctd);
                    otherwise
                        testCase.verifyFail(sprintf('%s must be thrown.',...
                            exceptionIdExpctd));
                end
            end
        end
        
        % Test
        function testUdHaarSplittingWarningReconstruction(testCase,...
                width,height,depth)
            
            % Parameters
            nLevels = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(nLevels-1)-1;
            nHorPad = 2^(nLevels-1)-1;
            nDepPad = 2^(nLevels-1)-1;
            srcImg = rand(height,width,depth);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer.NumberOfLevels = nLevels;
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:ReconstructionFailureException';
            messageExpctd = 'Failure occurs in reconstruction. Please check the split and padding size.';
            
            % Instantiation of target class
            import saivdr.restoration.*
            testCase.target = OlsOlaProcess3d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...             
                'PadSize',[nVerPad,nHorPad,nDepPad]);
            
            % Evaluation
            try
                step(testCase.target,srcImg);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                switch me.identifier
                    case exceptionIdExpctd
                        messageActual = me.message;
                        testCase.verifyEqual(messageActual, messageExpctd);
                    otherwise
                        testCase.verifyFail(sprintf('%s must be thrown.',...
                            exceptionIdExpctd));
                end
            end
        end
        
        % Test
        function testUdHaarIntegrityTestOff(testCase,width,height,depth)
            
            % Parameters
            nLevels = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(nLevels-1)-1;
            nHorPad = 2^(nLevels-1)-1;
            nDepPad = 2^(nLevels-1)-1;
            srcImg = rand(height,width,depth);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer.NumberOfLevels = nLevels;
            
            % Instantiation of target class
            import saivdr.restoration.*
            testCase.target = OlsOlaProcess3d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...             
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'IsIntegrityTest',false);
            
            % Evaluation
            try
                step(testCase.target,srcImg);
            catch me
                testCase.verifyFail(me.message);
            end
        end
        
        % Test
        function testSoftThresholding(testCase,width,height,depth,...
                useparallel)
            
            % Parameters
            nLevels = 3;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            nDepPad = 2^(nLevels-1);
            srcImg = rand(height,width,depth);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer.NumberOfLevels = nLevels;
            
            % Functions
            lambda = 1e-3;
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Expected values
            [coefspre,scales] = analyzer.step(srcImg);
            coefspst = g(coefspre,0);
            imgExpctd = synthesizer.step(coefspst,scales);
            
            % Instantiation of target class
            import saivdr.restoration.*
            coefsmanipulator = CoefsManipulator('Manipulation',g);
            testCase.target = OlsOlaProcess3d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'CoefsManipulator',coefsmanipulator,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...             
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Actual values
            imgActual = testCase.target.step(srcImg);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd));
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
        end
        
        % Test
        function testIterativeSoftThresholding(testCase,...
                width,height,depth,useparallel)
            
            % Parameters
            nIters = 5;
            nLevels = 3;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;            
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            nDepPad = 2^(nLevels-1);
            srcImg = rand(height,width,depth);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer.NumberOfLevels = nLevels;
            
            % Functions
            lambda = 1e-3;
            gamma  = 1e-3;
            f = @(x,xpre) testCase.softthresh(x,xpre,lambda,gamma);
            
            % Expected values
            h = srcImg;
            y = 0;
            for iIter = 1:nIters
                [v,scales] = analyzer.step(h);
                y = f(v,y);
                hu = synthesizer.step(y,scales);
                h = hu - srcImg;
            end
            imgExpctd = hu;
            
            % Instantiation of target class
            import saivdr.restoration.*
            coefsmanipulator = CoefsManipulator('Manipulation',f);
            testCase.target = OlsOlaProcess3d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'CoefsManipulator',coefsmanipulator,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...             
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Actual values
            h = srcImg;
            testCase.target.InitialState = 0;
            for iIter = 1:nIters
                hu = testCase.target.step(h);
                h = hu - srcImg;
            end
            imgActual = hu;
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd));
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
        end
        
        % Test
        function testIterativeSoftThresholdingInitialize(testCase,...
                width,height,depth,useparallel)
            
            % Parameters
            nIters = 5;
            nLevels = 3;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            nDepPad = 2^(nLevels-1);
            srcImg = rand(height,width,depth);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer.NumberOfLevels = nLevels;
            
            % Functions
            lambda = 1e-3;
            gamma  = 1e-3;
            f = @(x,xpre) testCase.softthresh(x,xpre,lambda,gamma);
            
            % Expected values
            h = srcImg;
            y = analyzer.step(h);
            for iIter = 1:nIters
                [v,scales] = analyzer.step(h);
                y = f(v,y);
                hu = synthesizer.step(y,scales);
                h = hu - srcImg;
            end
            imgExpctd = hu;
            
            % Instantiation of target class
            import saivdr.restoration.*
            coefsmanipulator = CoefsManipulator('Manipulation',f);
            testCase.target = OlsOlaProcess3d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'CoefsManipulator',coefsmanipulator,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...             
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Actual values
            h = srcImg;
            y = testCase.target.analyze(h);
            testCase.target.InitialState = y;
            for iIter = 1:nIters
                hu = testCase.target.step(h);
                h = hu - srcImg;
            end
            imgActual = hu;
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd));
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
        end        
        
        % Test
        function testIterativeSoftThresholdingInitializeCompare(testCase,...
                width,height,depth,isintegrity)
            
            % Parameters
            nIters = 5;
            nLevels = 3;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            nDepPad = 2^(nLevels-1);
            srcImg = rand(height,width,depth);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer.NumberOfLevels = nLevels;
            
            % Functions
            lambda = 1e-3;
            gamma  = 1e-3;
            f = @(x,xpre) testCase.softthresh(x,xpre,lambda,gamma);
            
            % Expected values
            h = srcImg;
            y = analyzer.step(h);
            for iIter = 1:nIters
                [v,scales] = analyzer.step(h);
                y = f(v,y);
                hu = synthesizer.step(y,scales);
                h = hu - srcImg;
            end
            imgExpctd = hu;
            coefsExpctd = y;
            scalesExpctd = scales;
            
            % Instantiation of target class
            import saivdr.restoration.*
            coefsmanipulator = CoefsManipulator('Manipulation',f);
            testCase.target = OlsOlaProcess3d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'CoefsManipulator',coefsmanipulator,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'IsIntegrityTest',isintegrity,...
                'UseParallel',true);
            serialProcess3d = OlsOlaProcess3d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'CoefsManipulator',coefsmanipulator,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...              
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'IsIntegrityTest',isintegrity,...                
                'UseParallel',false);
            
            % Actual values
            h = srcImg;
            yp = testCase.target.analyze(h); % Parallel 
            ys = serialProcess3d.analyze(h); % Serial
            for iSplit = 1:length(yp)
                qp = yp{iSplit};
                qs = ys{iSplit};
                for iCh = 1:length(qp)
                    testCase.verifyEqual(qp{iCh},qs{iCh},'AbsTol',1e-10);
                end
            end
            % Initalization with the same states
            testCase.target.InitialState = yp; % Parallel
            serialProcess3d.InitialState = yp; % Serial
            for iIter = 1:nIters
                hup = testCase.target.step(h); % Parallel step
                hus = serialProcess3d.step(h); % Serial step
                testCase.verifyEqual(hup,hus,'AbsTol',1e-10);
                h = hup - srcImg;
            end
            imgActual = hup;
            [coefsActual,scalesActual] = testCase.target.getCoefficients();
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd));
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            %
            testCase.verifySize(scalesActual,size(scalesExpctd));
            diff = max(abs(scalesExpctd(:) - scalesActual(:)));
            testCase.verifyEqual(scalesActual,scalesExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            %
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
        end
        
         % Test
        function testIterativeSoftThresholdingGpu(testCase,...
               useparallel,usegpu)
           
           if usegpu && gpuDeviceCount == 0
               warning('No GPU device was found.');
               return;
           end
            
            % Parameters
            nIters = 5;
            nLevels = 3;
            height_ = 32;
            width_ = 32;
            depth_ = 32;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            nDepPad = 2^(nLevels-1);
            srcImg = rand(height_,width_,depth_);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer.NumberOfLevels = nLevels;
            
            % Functions
            lambda = 1e-3;
            gamma  = 1e-3;
            f = @(x,xpre) testCase.softthresh(x,xpre,lambda,gamma);
            
            % Expected values
            h = srcImg;
            y = analyzer.step(h);
            for iIter = 1:nIters
                [v,scales] = analyzer.step(h);
                y = f(v,y);
                hu = synthesizer.step(y,scales);
                h = hu - srcImg;
            end
            imgExpctd = hu;
            
            % Instantiation of target class
            import saivdr.restoration.*
            coefsmanipulator = CoefsManipulator('Manipulation',f);
            testCase.target = OlsOlaProcess3d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'CoefsManipulator',coefsmanipulator,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...             
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel,...
                'UseGpu',usegpu);
            
            % Actual values
            h = srcImg;
            y = testCase.target.analyze(h);
            testCase.target.InitialState = y;
            for iIter = 1:nIters
                hu = testCase.target.step(h);
                h = hu - srcImg;
            end
            imgActual = hu;
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd));
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
        end        
        
    end
    
end

