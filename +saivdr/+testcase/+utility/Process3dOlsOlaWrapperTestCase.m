classdef Process3dOlsOlaWrapperTestCase < matlab.unittest.TestCase
    %PROCESS3DOLSOLAWRAPPERTESTCASE Test cases for Process3dOlsOlaWrapper  
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2018, Shogo MURAMATSU
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
        width = struct('small', 32, 'medium', 48, 'large', 64);
        height = struct('small', 32, 'medium', 48, 'large', 64);
        depth = struct('small', 32, 'medium', 48, 'large', 64);
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
    
    methods (Test)
        
        function testDefaultConstructor(testCase)
            
            % Expected values
            import saivdr.utility.*
            analyzerExpctd = [];
            synthesizerExpctd = [];
            
            % Instantiation
            testCase.target = Process3dOlsOlaWrapper();
            
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
            import saivdr.utility.*
            testCase.target = Process3dOlsOlaWrapper(...
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
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = Process3dOlsOlaWrapper(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer);
            
            % Actual values
            recImg = step(testCase.target,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(recImg,size(srcImg));
            diff = max(abs(srcImg(:) - recImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
         end
        
        % Test
        function testUdHaarSplitting(testCase,width,height,depth, level,useparallel)
            
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
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = Process3dOlsOlaWrapper(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Actual values
            recImg = step(testCase.target,srcImg,nLevels);

            % Evaluation
            testCase.verifySize(recImg,size(srcImg));
            diff = max(abs(srcImg(:) - recImg(:)));
            testCase.verifyEqual(recImg,srcImg,'AbsTol',1e-10,...
                sprintf('%g',diff));
        end  
        
        % Test
        function testUdHaarSplittingWarningFactor(testCase,width,height,depth,level,useparallel)
            
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
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalSplitFactorException';
            messageExpctd = 'Split factor must be a divisor of array size.';
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = Process3dOlsOlaWrapper(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Evaluation
            try
                step(testCase.target,srcImg,nLevels);
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
        function testUdHaarSplittingWarningReconstruction(testCase,width,height,depth)
            
            % Parameters
            level_ = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(level_-1)-1;
            nHorPad = 2^(level_-1)-1;
            nDepPad = 2^(level_-1)-1;
            srcImg = rand(height,width,depth);            
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();

            % Expected values
            exceptionIdExpctd = 'SaivDr:ReconstructionFailureException';
            messageExpctd = 'Failure occurs in reconstruction. Please check the split and padding size.';
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = Process3dOlsOlaWrapper(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad]);
            
            % Evaluation
            try
                step(testCase.target,srcImg,level_);
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
            level_ = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(level_-1)-1;
            nHorPad = 2^(level_-1)-1;
            nDepPad = 2^(level_-1)-1;
            srcImg = rand(height,width,depth);            
            import saivdr.dictionary.udhaar.*
            analyzer = UdHaarAnalysis3dSystem();
            synthesizer = UdHaarSynthesis3dSystem();


            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = Process3dOlsOlaWrapper(...
                'Analyzer',analyzer,...
                'Synthesizer',synthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'IsIntegrityTest',false);
            
            % Evaluation
            try
                step(testCase.target,srcImg,level_);
            catch me
                testCase.verifyFail(me.message);
            end
        end
        
    end
end
