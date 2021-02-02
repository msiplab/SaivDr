classdef MixtureOfUnitaryAnalysisSystemTestCase < matlab.unittest.TestCase
    %MIXTUREOfUnitaryANALYSISSYSTEMTESTCASE Test case for MixtureOfUnitaryAnalysisSystem
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2020, Shogo MURAMATSU
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
        function testTwoMixtureCase(testCase)
            
            % Parameter setting
            srcImg = rand(16,16);
            nLevelSet(1) = 3;
            nLevelSet(2) = 3;
            
            % Preparation
            import saivdr.dictionary.nsgenlotx.*
            import saivdr.dictionary.nsoltx.*
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'OutputMode','ParameterMatrixSet');
            subAnalyzers{1} = NsoltFactory.createAnalysis2dSystem(fb_,...
                'NumberOfLevels',nLevelSet(1));
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'TvmAngleInDegree',30,...
                'OutputMode','ParameterMatrixSet');
            subAnalyzers{2} = NsoltFactory.createAnalysis2dSystem(fb_,...
                'NumberOfLevels',nLevelSet(2));
            
            normFactor = 1/sqrt(length(subAnalyzers));
            
            % Expected value
            subScales = cell(2,1);
            [subCoefs{1}, subScales{1}] = step(subAnalyzers{1},srcImg);
            [subCoefs{2}, subScales{2}] = step(subAnalyzers{2},srcImg);
            coefsExpctd  = normFactor * cell2mat(subCoefs); 
            subScales{1} = [ subScales{1} ; -1 -1 ];
            subScales{2} = [ subScales{2} ]; 
            scalesExpctd = cell2mat(subScales);
                
            % Instantiation
            import saivdr.dictionary.mixture.*
            testCase.analyzer = MixtureOfUnitaryAnalysisSystem(...
                'UnitaryAnalyzerSet',subAnalyzers);
            
            % Actual value
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:)-coefsActual(:))...
                ./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));

        end
        
        % Test
        function testThreeMixtureCase(testCase)
            
            % Parameter setting
            srcImg = rand(16,16);
            nLevelSet(1) = 1;
            nLevelSet(2) = 2;
            nLevelSet(3) = 3;
            
            % Preparation
            import saivdr.dictionary.nsgenlotx.*
            import saivdr.dictionary.nsoltx.*
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'OutputMode','ParameterMatrixSet');
            subAnalyzers{1} = NsoltFactory.createAnalysis2dSystem(fb_,...
                'NumberOfLevels',nLevelSet(1));
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'TvmAngleInDegree',30,...
                'OutputMode','ParameterMatrixSet');
            subAnalyzers{2} = NsoltFactory.createAnalysis2dSystem(fb_,...
                'NumberOfLevels',nLevelSet(2));
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'TvmAngleInDegree',60,...
                'OutputMode','ParameterMatrixSet');
            subAnalyzers{3} = NsoltFactory.createAnalysis2dSystem(fb_,...
                'NumberOfLevels',nLevelSet(3));
            
            normFactor = 1/sqrt(length(subAnalyzers));
            
            % Expected value
            subScales = cell(3,1);
            [subCoefs{1}, subScales{1}] = step(subAnalyzers{1},srcImg);
            [subCoefs{2}, subScales{2}] = step(subAnalyzers{2},srcImg);
            [subCoefs{3}, subScales{3}] = step(subAnalyzers{3},srcImg);
            coefsExpctd  = normFactor * cell2mat(subCoefs); 
            subScales{1} = [ subScales{1} ; -1 -1 ];
            subScales{2} = [ subScales{2} ; -1 -1 ];
            subScales{3} = [ subScales{3} ];
            scalesExpctd = cell2mat(subScales);
            
            % Instantiation
            import saivdr.dictionary.mixture.*
            testCase.analyzer = MixtureOfUnitaryAnalysisSystem(...
                'UnitaryAnalyzerSet',subAnalyzers);
            
            % Actual value
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:)-coefsActual(:))...
                ./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));

        end  
        
        % Test
        function testFiveMixtureCase(testCase)
            
            % Parameter setting
            srcImg = rand(16,16);
            nDics = 5;
            nLevelSet = zeros(nDics,1);
            for iDic = 1:nDics
                nLevelSet(iDic) = 3;
            end
            
            % Preparation
            import saivdr.dictionary.nsgenlotx.*
            import saivdr.dictionary.nsoltx.*
            subAnalyzers = cell(nDics,1);
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'OutputMode','ParameterMatrixSet');
            subAnalyzers{1} = NsoltFactory.createAnalysis2dSystem(fb_);
            phi = [ -30 30 60 120 ];
            for iDic = 2:nDics
                fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                    'NumberOfVanishingMoments',2,...
                    'TvmAngleInDegree',phi(iDic-1),...
                    'OutputMode','ParameterMatrixSet');
                subAnalyzers{iDic} = NsoltFactory.createAnalysis2dSystem(fb_,...
                    'NumberOfLevels',nLevelSet(iDic));
            end
            
            normFactor = 1/sqrt(length(subAnalyzers));
            
            % Expected value
            subCoefs = cell(1,nDics);
            subScales = cell(nDics,1);
            for iDic = 1:nDics-1
                [subCoefs{iDic}, tmpScales] = ...
                    step(subAnalyzers{iDic},srcImg);
                subScales{iDic} = ...
                    [ tmpScales ; -1 -1 ];
            end
            iDic = nDics;
            [subCoefs{iDic}, tmpScales] = ...
                step(subAnalyzers{iDic},srcImg);
            subScales{iDic} = tmpScales;
            coefsExpctd  = normFactor * cell2mat(subCoefs); 
            scalesExpctd = cell2mat(subScales);

            % Instantiation
            import saivdr.dictionary.mixture.*
            testCase.analyzer = MixtureOfUnitaryAnalysisSystem(...
                'UnitaryAnalyzerSet',subAnalyzers);
            
            % Actual value
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:)-coefsActual(:))...
                ./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));

        end        
        
        % Test
        function testClone(testCase)
            
            % Parameter setting
            srcImg = rand(16,16);
            nDics = 5;
            nLevelSet = zeros(nDics,1);
            for iDic = 1:nDics
                nLevelSet(iDic) = 3;
            end
            
            % Preparation
            import saivdr.dictionary.nsgenlotx.*
            import saivdr.dictionary.nsoltx.*
            subAnalyzers = cell(nDics,1);
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'OutputMode','ParameterMatrixSet');
            subAnalyzers{1} = NsoltFactory.createAnalysis2dSystem(fb_,...
                'NumberOfLevels',nLevelSet(1));
            phi = [ -30 30 60 120 ];
            for iDic = 2:nDics
                fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                    'NumberOfVanishingMoments',2,...
                    'TvmAngleInDegree',phi(iDic-1),...
                    'OutputMode','ParameterMatrixSet');
                subAnalyzers{iDic} = NsoltFactory.createAnalysis2dSystem(fb_,...
                    'NumberOfLevels',nLevelSet(iDic));
            end

            % Instantiation
            import saivdr.dictionary.mixture.*
            testCase.analyzer = MixtureOfUnitaryAnalysisSystem(...
                'UnitaryAnalyzerSet',subAnalyzers);
            
            % Clone 
            cloneAnalyzer = clone(testCase.analyzer);
            
            % Evaluation
            testCase.verifyEqual(cloneAnalyzer,testCase.analyzer);
            testCase.verifyFalse(cloneAnalyzer == testCase.analyzer);
            prpOrg = get(testCase.analyzer,'UnitaryAnalyzerSet');
            prpCln = get(cloneAnalyzer,'UnitaryAnalyzerSet');
            testCase.verifyEqual(prpCln,prpOrg);
            for iDic = 1:nDics
                testCase.verifyFalse(prpCln{iDic} == prpOrg{iDic});
            end
            %
            [coefExpctd,scaleExpctd] = step(testCase.analyzer,srcImg);
            [coefActual,scaleActual] = step(cloneAnalyzer,srcImg);
            testCase.assertEqual(coefActual,coefExpctd);
            testCase.assertEqual(scaleActual,scaleExpctd);
            
        end                
        

    end
end
