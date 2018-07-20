classdef Analysis3dOlsWrapperTestCase < matlab.unittest.TestCase
    %ANALYSIS3DOLSWRAPPERTESTCASE Test case for Analysis3dOlsWrapper
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
        analyzer
    end
    
    methods (TestMethodTeardown)
        function deteleObject(testCase)
            delete(testCase.analyzer);
        end
    end
    
    methods (Test)
        
        % Test
        
        function testDefaultConstruction(testCase)     
            
            % Expected values
            import saivdr.dictionary.olaols.*
            analyzerExpctd = [];
            boundaryOperationExpctd = [];
            
            % Instantiation
            testCase.analyzer = Analysis3dOlsWrapper();
            
            % Actual value
            analyzerActual = testCase.analyzer.Analyzer;
            boundaryOperationActual = testCase.analyzer.BoundaryOperation;
            
            % Evaluation
            testCase.assertEqual(analyzerActual,analyzerExpctd);
            testCase.assertEqual(boundaryOperationActual,boundaryOperationExpctd);  
        end

        
        % Test
        function testAnalyzer(testCase)
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            analyzerExpctd = UdHaarAnalysis3dSystem();

            % Instantiation
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',analyzerExpctd);
            
            % Actual value
            analyzerActual = get(testCase.analyzer,'Analyzer');
            
            % Evaluation
            testCase.assertEqual(analyzerActual,analyzerExpctd);

        end        

        % Test
        function testUdHaar(testCase,height,width,depth,level)

            % Parameters
            nLevels = level;
            srcImg = rand(height,width,depth);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem();
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg,nLevels);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',refAnalyzer);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
        end
    
        
        % Test
        function testUdHaarSplitting(testCase,width,height,depth,level,useparallel)
            
            % Parameters
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            nDepPad = 2^(level-1);
            srcImg = rand(height,width,depth);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem();
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg,level);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,level);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
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
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:ReconstructionFailureException';
            messageExpctd = 'Failure occurs in reconstruction. Please check the split and padding size.';
            
            % Preparation
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem();
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',false);
            
            % Evaluation
            try
                step(testCase.analyzer,srcImg,level_);
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
        function testUdHaarSplittingWarningFactor(testCase,width,height,depth,level)
            
            % Parameters
            nVerSplit = 3;
            nHorSplit = 3;
            nDepSplit = 3;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            nDepPad = 2^(level-1);
            srcImg = rand(height,width,depth);
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalSplitFactorException';
            messageExpctd = 'Split factor must be a divisor of array size.';
            
            % Preparation
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem();
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',false);
            
            % Evaluation
            try
                step(testCase.analyzer,srcImg,level);
                if mod(width,nHorSplit) ~=0 || ...
                    mod(height,nVerSplit) ~= 0 || ...
                     mod(depth,nDepSplit) ~=0
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
    
    end
end
