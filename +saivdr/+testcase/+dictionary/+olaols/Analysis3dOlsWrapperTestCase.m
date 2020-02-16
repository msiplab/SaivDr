classdef Analysis3dOlsWrapperTestCase < matlab.unittest.TestCase
    %ANALYSIS3DOLSWRAPPERTESTCASE Test case for Analysis3dOlsWrapper
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
        width = struct('small', 32, 'medium', 48, 'large', 64);
        height = struct('small', 32, 'medium', 48, 'large', 64);
        depth = struct('small', 32, 'medium', 48, 'large', 64);
        vsplit = struct('small', 1, 'medium', 2, 'large', 4);
        hsplit = struct('small', 1, 'medium', 2, 'large', 4);        
        dsplit = struct('small', 1, 'medium', 2, 'large', 4);                        
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
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',nLevels);
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',refAnalyzer);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
        end
        
        
        % Test
        function testUdHaarCellOutput(testCase,height,width,depth,level)

            % Parameters
            nLevels = level;
            srcImg = rand(height,width,depth);
            nSplit = 1;
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',nLevels);
            [coefs,scales] = step(refAnalyzer,srcImg);
            coefsExpctd = cell(nSplit,1);
            coefsExpctd{1} = coefs;
            scalesExpctd = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'OutputType','Cell');
            
            % Actual values
            [coefsActual, scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual{1},size(coefsExpctd{1}));
            diff = max(abs(coefsExpctd{1}(:) - coefsActual{1}(:)));
            testCase.verifyEqual(coefsActual{1},coefsExpctd{1},...
                'AbsTol',1e-10,sprintf('%g',diff));
        end
    
        
        % Test
        function testUdHaarSplittingSize(testCase,width,height,depth,...
                useparallel)
            
            % Parameters
            nLevels = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            nDepPad = 2^(nLevels-1);
            srcImg = rand(height,width,depth);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',nLevels);
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg);
            
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
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
        end
        
        % Test
        function testUdHaarSplittingSplit(testCase,vsplit,hsplit,dsplit,...
                useparallel)
            
            % Parameters
            nLevels = 2;
            height_ = 48;
            width_ = 48;
            depth_ = 48;
            nVerSplit = vsplit;
            nHorSplit = hsplit;
            nDepSplit = dsplit;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            nDepPad = 2^(nLevels-1);
            srcImg = rand(height_,width_,depth_);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',nLevels);
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg);
            
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
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
        end
        
        % Test
        function testUdHaarSplittingLevel(testCase,level,useparallel)
            
            % Parameters
            height_ = 48;
            width_ = 48;
            depth_ = 48;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            nDepPad = 2^(level-1);
            srcImg = rand(height_,width_,depth_);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',level);
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg);
            
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
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
        end        
        
        % Test
        function testUdHaarSplittingCellOutputSize(testCase,width,height,depth,...
                useparallel)
            
            % Parameters
            nLevels = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            nDepPad = 2^(nLevels-1);
            srcImg = rand(height,width,depth);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',nLevels);
            [coefs,scales] = step(refAnalyzer,srcImg);
            nSplit = nVerSplit*nHorSplit*nDepSplit;
            [coefsExpctd, scalesExpctd] = testCase.splitCoefs_(...
                coefs,scales,[nVerSplit nHorSplit nDepSplit]);

            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel,...
                'OutputType','Cell');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            for iSplit = 1:nSplit
                testCase.verifySize(coefsActual{iSplit},size(coefsExpctd{iSplit}));
                diff = max(abs(coefsExpctd{iSplit}(:) - coefsActual{iSplit}(:)));
                testCase.verifyEqual(coefsActual{iSplit},coefsExpctd{iSplit},...
                    'AbsTol',1e-10,sprintf('%g',diff));            
            end
        end        

        % Test
        function testUdHaarSplittingCellOutputSplit(testCase,vsplit,hsplit,dsplit,...
                useparallel)
            
            % Parameters
            nLevels = 2;
            height_ = 48;
            width_ = 48;
            depth_ = 48;
            nVerSplit = vsplit;
            nHorSplit = hsplit;
            nDepSplit = dsplit;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            nDepPad = 2^(nLevels-1);
            srcImg = rand(height_,width_,depth_);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',nLevels);
            [coefs,scales] = step(refAnalyzer,srcImg);
            nSplit = nVerSplit*nHorSplit*nDepSplit;
            [coefsExpctd, scalesExpctd] = testCase.splitCoefs_(...
                coefs,scales,[nVerSplit nHorSplit nDepSplit]);

            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel,...
                'OutputType','Cell');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            for iSplit = 1:nSplit
                testCase.verifySize(coefsActual{iSplit},size(coefsExpctd{iSplit}));
                diff = max(abs(coefsExpctd{iSplit}(:) - coefsActual{iSplit}(:)));
                testCase.verifyEqual(coefsActual{iSplit},coefsExpctd{iSplit},...
                    'AbsTol',1e-10,sprintf('%g',diff));            
            end
        end        

        % Test
        function testUdHaarSplittingCellOutputLevel(testCase,level,useparallel)
            
            % Parameters
            height_ = 48;
            width_ = 48;
            depth_ = 48;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            nDepPad = 2^(level-1);
            srcImg = rand(height_,width_,depth_);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',level);
            [coefs,scales] = step(refAnalyzer,srcImg);
            nSplit = nVerSplit*nHorSplit*nDepSplit;
            [coefsExpctd, scalesExpctd] = testCase.splitCoefs_(...
                coefs,scales,[nVerSplit nHorSplit nDepSplit]);

            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel,...
                'OutputType','Cell');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            for iSplit = 1:nSplit
                testCase.verifySize(coefsActual{iSplit},size(coefsExpctd{iSplit}));
                diff = max(abs(coefsExpctd{iSplit}(:) - coefsActual{iSplit}(:)));
                testCase.verifyEqual(coefsActual{iSplit},coefsExpctd{iSplit},...
                    'AbsTol',1e-10,sprintf('%g',diff));            
            end
        end        
        
        
        
        % Test
        function testUdHaarSplittingWarningReconstruction(testCase,width,height,depth)
            
            % Parameters
            nLevels = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(nLevels-1)-1;
            nHorPad = 2^(nLevels-1)-1;
            nDepPad = 2^(nLevels-1)-1;
            srcImg = rand(height,width,depth);
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:ReconstructionFailureException';
            messageExpctd = 'Failure occurs in reconstruction. Please check the split and padding size.';
            
            % Preparation
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',nLevels);
            
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
                step(testCase.analyzer,srcImg);
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
        function testUdHaarSplittingIntegrityTestOff(testCase,width,height,depth)
            
            % Parameters
            nLevels = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(nLevels-1)-1;
            nHorPad = 2^(nLevels-1)-1;
            nDepPad = 2^(nLevels-1)-1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',nLevels);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',false,...
                'IsIntegrityTest',false);
            
            % Evaluation
            try
                step(testCase.analyzer,srcImg);
            catch me
                testCase.verifyFail(me.message);
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
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',level);
            
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
                step(testCase.analyzer,srcImg);
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
    
         % Test
        function testUdHaarSplitFactor(testCase,width,height,depth,level,useparallel)
            
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
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',level);
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
        end
        
        % Test
        function testUdHaarSplitFactorWarning(testCase,width,height,depth,level)
            
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
            refAnalyzer = UdHaarAnalysis3dSystem('NumberOfLevels',level);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis3dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',false);
            
            % Evaluation
            try
                step(testCase.analyzer,srcImg);
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
    
     methods (Static, Access = private) 
        
        function [coefsCrop, scalesCrop] = splitCoefs_(coefs,scales,splitFactor)
            import saivdr.dictionary.utility.Direction
            nChs = size(scales,1);
            nSplit = prod(splitFactor);
            nVerSplit = splitFactor(Direction.VERTICAL);
            nHorSplit = splitFactor(Direction.HORIZONTAL);
            nDepSplit = splitFactor(Direction.DEPTH);
            %
            coefsCrop = cell(nSplit,1);
            for iSplit = 1:nSplit
                coefsCrop{iSplit} = [];
            end
            scalesCrop = zeros(nChs,3);
            %
            eIdx = 0;
            for iCh = 1:nChs
                sIdx = eIdx + 1;
                eIdx = sIdx + prod(scales(iCh,:)) - 1;
                nRows = scales(iCh,Direction.VERTICAL);
                nCols = scales(iCh,Direction.HORIZONTAL);
                nLays = scales(iCh,Direction.DEPTH);
                coefArrays = reshape(coefs(sIdx:eIdx),[nRows nCols nLays]);
                %
                nSubRows = nRows/nVerSplit;
                nSubCols = nCols/nHorSplit;
                nSubLays = nLays/nDepSplit;
                iSplit = 0;
                for iDepSplit = 1:nDepSplit
                    sLayIdx = (iDepSplit-1)*nSubLays + 1;
                    eLayIdx = iDepSplit*nSubLays;
                    for iHorSplit = 1:nHorSplit
                        sColIdx = (iHorSplit-1)*nSubCols + 1;
                        eColIdx = iHorSplit*nSubCols;
                        for iVerSplit = 1:nVerSplit
                            sRowIdx = (iVerSplit-1)*nSubRows + 1;
                            eRowIdx = iVerSplit*nSubRows;
                            subCoefArrays = coefArrays(...
                                sRowIdx:eRowIdx,sColIdx:eColIdx,sLayIdx:eLayIdx);
                            %
                            iSplit = iSplit + 1;
                            coefsCrop{iSplit} = [coefsCrop{iSplit} subCoefArrays(:).'];
                        end
                    end
                end
                scalesCrop(iCh,:) = [nSubRows nSubCols nSubLays];
            end
        end
     end
    
end
