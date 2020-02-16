classdef OlsOlaProcess2dTestCase < matlab.unittest.TestCase
    %OLSOLAPROCESS2DTESTCASE Test cases for OlsOlaProcess2d  
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
        useparallel = { true, false };
        width = struct('small', 64, 'medium', 96, 'large', 128);
        height = struct('small', 64, 'medium', 96, 'large', 128);
        vsplit = struct('small', 1, 'medium', 2, 'large', 4); 
        hsplit = struct('small', 1, 'medium', 2, 'large', 4);         
        level = struct('flat',1, 'sharrow',3,'deep', 5);
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
        function y = softthresh(x,xpre,lambda,gamma)
            u = xpre-gamma*x;
            v = abs(u)-lambda;
            y = sign(u).*(v+abs(v))/2;
        end        
        function [v,x] = coefpdshshc(t,xpre,lambda,gamma)
            u = xpre-gamma*t;
            w = abs(u)-lambda;
            x = sign(u).*(w+abs(w))/2;
            v = 2*x - xpre;
        end
    end
    
    methods (Test)
        
        function testDefaultConstructor(testCase)
            
            % Expected values
            import saivdr.utility.*
            analyzerExpctd = [];
            synthesizerExpctd = [];            
            
            % Instantiation
            testCase.target = OlsOlaProcess2d();
            
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
            analyzerExpctd = UdHaarAnalysis2dSystem();
            synthesizerExpctd = UdHaarSynthesis2dSystem();
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = OlsOlaProcess2d(...
                'Analyzer',analyzerExpctd,...
                'Synthesizer',synthesizerExpctd);
            
            % Actual value
            analyzerActual = get(testCase.target,'Analyzer');
            synthesizerActual = get(testCase.target,'Synthesizer');
            
            % Evaluation
            testCase.assertEqual(analyzerActual,analyzerExpctd);
            testCase.assertEqual(synthesizerActual,synthesizerExpctd);
            
        end
        
        function testUdHaar(testCase,height,width,level)
            
            % Parameters
            nLevels = level;
            srcImg = rand(height,width);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            synthesizer = UdHaarSynthesis2dSystem();
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = OlsOlaProcess2d(...
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
        function testUdHaarSplittingSize(testCase,width,height,level,useparallel)
            
            % Parameters
            nLevels = level;
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            srcImg = rand(height,width);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            synthesizer = UdHaarSynthesis2dSystem();
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = OlsOlaProcess2d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
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
        function testUdHaarSplittingSplit(testCase,vsplit,hsplit,level,useparallel)
            
            % Parameters
            nLevels = level;
            height_ = 96;
            width_ = 96;
            nVerSplit = vsplit;
            nHorSplit = hsplit;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            srcImg = rand(height_,width_);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            synthesizer = UdHaarSynthesis2dSystem();
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = OlsOlaProcess2d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
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
        function testUdHaarSplittingWarningFactor(testCase,width,height,level,useparallel)
            
            % Parameters
            nLevels = level;
            nVerSplit = 3;
            nHorSplit = 3;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            srcImg = rand(height,width);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            synthesizer = UdHaarSynthesis2dSystem();
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalSplitFactorException';
            messageExpctd = 'Split factor must be a divisor of array size.';
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = OlsOlaProcess2d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
                'UseParallel',useparallel);
            
            % Evaluation
            try
                step(testCase.target,srcImg);
                if mod(width,nHorSplit) ~=0 || mod(height,nVerSplit) ~= 0
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
        function testUdHaarSplittingWarningReconstruction(testCase,width,height)
            
            % Parameters
            level_ = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(level_-1)-1;
            nHorPad = 2^(level_-1)-1;
            srcImg = rand(height,width);            
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis2dSystem('NumberOfLevels',level_);
            synthesizer = UdHaarSynthesis2dSystem();

            % Expected values
            exceptionIdExpctd = 'SaivDr:ReconstructionFailureException';
            messageExpctd = 'Failure occurs in reconstruction. Please check the split and padding size.';
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = OlsOlaProcess2d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad]);
            
            % Evaluation
            try
                testCase.target.step(srcImg);
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
        function testUdHaarIntegrityTestOff(testCase,width,height)
            
            % Parameters
            level_ = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(level_-1)-1;
            nHorPad = 2^(level_-1)-1;
            srcImg = rand(height,width);            
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis2dSystem('NumberOfLevels',level_);
            synthesizer = UdHaarSynthesis2dSystem();

            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = OlsOlaProcess2d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
                'IsIntegrityTest',false);
            
            % Evaluation
            try
                testCase.target.step(srcImg);
            catch me
                testCase.verifyFail(me.message);
            end
        end
        
        % Test
        function testSoftThresholding(testCase,width,height,useparallel)
            
            % Parameters
            nLevels = 3;
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            srcImg = rand(height,width);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            synthesizer = UdHaarSynthesis2dSystem();
            
            % Functions
            lambda = 1e-3;
            g = @(x) sign(x).*((abs(x)-lambda)+abs(abs(x)-lambda))/2;

            % Expected values
            [coefspre,scales] = analyzer.step(srcImg);
            coefspst = g(coefspre);
            imgExpctd = synthesizer.step(coefspst,scales);
            
            % Instantiation of target class
            import saivdr.utility.*            
            coefsmanipulator = CoefsManipulator(...
                'Manipulation',g);
            testCase.target = OlsOlaProcess2d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'CoefsManipulator',coefsmanipulator,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
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
        function testIterativeSoftThresholding(testCase,width,height,useparallel)
            
            % Parameters
            nIters = 5;
            nLevels = 3;
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            srcImg = rand(height,width);
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            synthesizer = UdHaarSynthesis2dSystem();
            
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
            import saivdr.utility.*
            coefsmanipulator = CoefsManipulator(...
                'Manipulation',f,...
                'IsFeedBack',true);
            testCase.target = OlsOlaProcess2d(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'CoefsManipulator',coefsmanipulator,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
                'UseParallel',useparallel);
            
            % Actual values
            h = srcImg;
            coefsmanipulator.State = 0;
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
