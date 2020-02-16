classdef Analysis2dOlsWrapperTestCase < matlab.unittest.TestCase
    %ANALYSIS2DOLSWRAPPERTESTCASE Test case for Analysis2dOlsWrapper
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
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
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
        
        %Test
        function testUdHaarCellOutput(testCase,height,width,level)
            
            % Parameters
            nLevels = level;
            srcImg = rand(height,width);
            nSplit = 1;
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            [coefs,scales] = step(refAnalyzer,srcImg);
            coefsExpctd = cell(nSplit,1);
            coefsExpctd{1} = coefs;
            scalesExpctd = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'OutputType','Cell');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual{1},size(coefsExpctd{1}));
            diff = max(abs(coefsExpctd{1}(:) - coefsActual{1}(:)));
            testCase.verifyEqual(coefsActual{1},coefsExpctd{1},...
                'AbsTol',1e-10, sprintf('%g',diff));
        end
        
        % Test
        function testUdHaarSplittingSize(testCase,width,height,useparallel)
            
            % Parameters
            nLevels = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            srcImg = rand(height,width);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
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
        
        function testUdHaarSplittingSplit(testCase,vsplit,hsplit,useparallel)
            
            % Parameters
            height_ = 96;
            width_ = 96;
            nLevels = 2;
            nVerSplit = vsplit;
            nHorSplit = hsplit;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            srcImg = rand(height_,width_);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
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
            height_ = 96;
            width_ = 96;
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            srcImg = rand(height_,width_);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',level);
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
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
        function testUdHaarSplittingCellOutputSize(testCase,width,height,...
                useparallel)
            
            % Parameters
            nLevels = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            srcImg = rand(height,width);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            [coefs,scales] = step(refAnalyzer,srcImg);
            nSplit = nVerSplit*nHorSplit;
            [coefsExpctd, scalesExpctd] = testCase.splitCoefs_(...
                coefs,scales,[nVerSplit nHorSplit]);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
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
                    'AbsTol',1e-10, sprintf('%g',diff));
            end
        end
        
        % Test
        function testUdHaarSplittingCellOutputSplit(testCase,vsplit,hsplit,...
                useparallel)
            
            % Parameters
            nLevels =  2;
            height_ = 96;
            width_ = 96;
            nVerSplit = vsplit;
            nHorSplit = hsplit;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            srcImg = rand(height_,width_);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            [coefs,scales] = step(refAnalyzer,srcImg);
            nSplit = nVerSplit*nHorSplit;
            [coefsExpctd, scalesExpctd] = testCase.splitCoefs_(...
                coefs,scales,[nVerSplit nHorSplit]);

            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
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
                    'AbsTol',1e-10, sprintf('%g',diff));
            end
        end
        
        % Test
        function testUdHaarSplittingCellOutputLevel(testCase,level,useparallel)
            
            % Parameters
            height_ = 96;
            width_ = 96;
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            srcImg = rand(height_,width_);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',level);
            [coefs,scales] = step(refAnalyzer,srcImg);
            nSplit = nVerSplit*nHorSplit;
            [coefsExpctd, scalesExpctd] = testCase.splitCoefs_(...
                coefs,scales,[nVerSplit nHorSplit]);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
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
                    'AbsTol',1e-10, sprintf('%g',diff));
            end
        end
        
        
        % Test
        function testUdHaarSplittingWarningReconstruction(testCase,width,height)
            
            % Parameters
            nLevels = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(nLevels-1)-1;
            nHorPad = 2^(nLevels-1)-1;
            srcImg = rand(height,width);
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:ReconstructionFailureException';
            messageExpctd = 'Failure occurs in reconstruction. Please check the split and padding size.';
            
            % Preparation
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
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
        function testUdHaarSplittingIntegrityTestOff(testCase,width,height)
            
            % Parameters
            nLevels = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(nLevels-1)-1;
            nHorPad = 2^(nLevels-1)-1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
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
        function testUdHaarSplittingWarningFactor(testCase,width,height,level)
            
            % Parameters
            nVerSplit = 3;
            nHorSplit = 3;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            srcImg = rand(height,width);
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalSplitFactorException';
            messageExpctd = 'Split factor must be a divisor of array size.';
            
            % Preparation
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',level);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'PadSize',[nVerPad,nHorPad],...
                'UseParallel',false);
            
            % Evaluation
            try
                step(testCase.analyzer,srcImg);
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
        function testUdHaarSplitFactor(testCase,width,height,level,useparallel)
            
            % Parameters
            nVerSplit = 2;
            nHorSplit = 2;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            srcImg = rand(height,width);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',level);
            [coefsExpctd,scalesExpctd] = step(refAnalyzer,srcImg);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'SplitFactor',[nVerSplit,nHorSplit],...
                'PadSize',[nVerPad,nHorPad],...
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
        function testUdHaarSplitFactorWarning(testCase,width,height,level)
            
            % Parameters
            nVerSplit = 3;
            nHorSplit = 3;
            nVerPad = 2^(level-1);
            nHorPad = 2^(level-1);
            srcImg = rand(height,width);
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalSplitFactorException';
            messageExpctd = 'Split factor must be a divisor of array size.';
            
            % Preparation
            import saivdr.dictionary.udhaar.*
            refAnalyzer = UdHaarAnalysis2dSystem('NumberOfLevels',level);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.analyzer = Analysis2dOlsWrapper(...
                'Analyzer',refAnalyzer,...
                'SplitFactor',[nVerSplit,nHorSplit],...
                'PadSize',[nVerPad,nHorPad],...
                'UseParallel',false);
            
            % Evaluation
            try
                step(testCase.analyzer,srcImg);
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
        
    end
    
    methods (Static, Access = private)
        
        function [coefsCrop, scalesCrop] = splitCoefs_(coefs,scales,splitFactor)
            import saivdr.dictionary.utility.Direction
            nChs = size(scales,1);
            nSplit = prod(splitFactor);
            nVerSplit = splitFactor(Direction.VERTICAL);
            nHorSplit = splitFactor(Direction.HORIZONTAL);
            %
            coefsCrop = cell(nSplit,1);
            for iSplit = 1:nSplit
                coefsCrop{iSplit} = [];
            end
            scalesCrop = zeros(nChs,2);
            %
            eIdx = 0;
            for iCh = 1:nChs
                sIdx = eIdx + 1;
                eIdx = sIdx + prod(scales(iCh,:)) - 1;
                nRows = scales(iCh,Direction.VERTICAL);
                nCols = scales(iCh,Direction.HORIZONTAL);
                coefArrays = reshape(coefs(sIdx:eIdx),[nRows nCols]);
                %
                nSubRows = nRows/nVerSplit;
                nSubCols = nCols/nHorSplit;
                iSplit = 0;
                for iHorSplit = 1:nHorSplit
                    sColIdx = (iHorSplit-1)*nSubCols + 1;
                    eColIdx = iHorSplit*nSubCols;
                    for iVerSplit = 1:nVerSplit
                        sRowIdx = (iVerSplit-1)*nSubRows + 1;
                        eRowIdx = iVerSplit*nSubRows;
                        subCoefArrays = coefArrays(sRowIdx:eRowIdx,sColIdx:eColIdx);
                        %
                        iSplit = iSplit + 1;
                        coefsCrop{iSplit} = [coefsCrop{iSplit} subCoefArrays(:).'];
                    end
                end
                scalesCrop(iCh,:) = [nSubRows nSubCols];
            end
        end
    end
end