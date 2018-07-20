classdef Analysis2dOlsWrapperTestCase < matlab.unittest.TestCase
    %ANALYSIS2DOLSWRAPPERTESTCASE Test case for Analysis2dOlsWrapper
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
        width = struct('small', 64, 'medium', 96, 'large', 128);
        height = struct('small', 64, 'medium', 96, 'large', 128);
        level = struct('flat',1, 'sharrow',3,'deep', 5);
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
            testCase.analyzer = Analysis2dOlsWrapper();
            
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
            analyzerExpctd = UdHaarAnalysis2dSystem();

            % Instantiation
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',analyzerExpctd);
            
            % Actual value
            analyzerActual = get(testCase.analyzer,'Analyzer');
            
            % Evaluation
            testCase.assertEqual(analyzerActual,analyzerExpctd);

        end        

        % Test
        function testUdHaar(testCase,height,width,level)

            % Parameters
            nLevels = level;
            srcImg = rand(height,width);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem();
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg,nLevels);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
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
        function testUdHaarSplitting(testCase,width,height,level)
            
            % Parameters
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            srcImg = rand(height,width);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem();
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg,level);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
                'UseParallel',false);
            
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
        
    end
    
end
