classdef TypeIIAnalysisSystemTestCase < matlab.unittest.TestCase
    %TYPEIIANALYSISSYSTEMTESTCASE Test case for TypeIIAnalysisSystem
    %
    % SVN identifier:
    % $Id: TypeIIAnalysisSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2015, Shogo MURAMATSU
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
        function testDefaultConstruction(testCase)
            
            % Expected values
            import saivdr.dictionary.nsolt.*
            lppufbExpctd = OvsdLpPuFb2dTypeIIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            testCase.analyzer = TypeIIAnalysisSystem();
            
            % Actual value
            lppufbActual = get(testCase.analyzer,'LpPuFb2d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end
        
        % Test
        function testDefaultConstruction6plus2(testCase)
            
            % Preperation
            nChs = [6 2];
            
            % Expected values
            import saivdr.dictionary.nsolt.*
            lppufbExpctd = OvsdLpPuFb2dTypeIIVm1System(...
                'NumberOfChannels',nChs,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            import saivdr.dictionary.nsolt.ChannelGroup
            testCase.analyzer = TypeIIAnalysisSystem(...
                'NumberOfSymmetricChannels',nChs(ChannelGroup.UPPER),...
                'NumberOfAntisymmetricChannels',nChs(ChannelGroup.LOWER));
            
            % Actual value
            lppufbActual = get(testCase.analyzer,'LpPuFb2d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end
        
        % Test
        function testStepDec11Ch32Ord00Level1PeriodicExtVm0(testCase)
            
            dec = 1;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec11Ch32Ord00Level1PeriodicExtVm1(testCase)
            
            dec = 1;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec11Ch32Ord00Level2PeriodicExtVm0(testCase)
            
            dec = 1;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels', decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec11Ch32Ord00Level2PeriodicExtVm1(testCase)
            
            dec = 1;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels', decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch32Ord00Level1(testCase)
            
            dec = 2;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch32Ord00Level2eriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 3 2];
            ch = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch43Ord00Level1(testCase)
            
            dec = 2;
            decch = [ dec dec 4 3 ];
            nChs = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch43Ord00Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 4 3 ];
            ch = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv1{2};
            coefs{9} = coefsExpctdLv1{3};
            coefs{10} = coefsExpctdLv1{4};
            coefs{11} = coefsExpctdLv1{5};
            coefs{12} = coefsExpctdLv1{6};
            coefs{13} = coefsExpctdLv1{7};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
            
        end
        
        % Test
        function testStepDec22Ch54Ord00Level1(testCase)
            
            dec = 2;
            decch = [ dec dec 5 4 ];
            nChs= sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch54Ord00Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 5 4 ];
            ch = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv2{8};
            coefs{9} = coefsExpctdLv2{9};
            coefs{10} = coefsExpctdLv1{2};
            coefs{11} = coefsExpctdLv1{3};
            coefs{12} = coefsExpctdLv1{4};
            coefs{13} = coefsExpctdLv1{5};
            coefs{14} = coefsExpctdLv1{6};
            coefs{15} = coefsExpctdLv1{7};
            coefs{16} = coefsExpctdLv1{8};
            coefs{17} = coefsExpctdLv1{9};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
            
        end
        
        % Test
        function testStepDec11Ch32Ord22Level1(testCase)
            
            dec = 1;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec11Ch32Ord22Level2PeriodicExt(testCase)
            
            dec = 1;
            decch = [ dec dec 3 2 ];
            ch = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
            
        end
        
        % Test
        function testStepDec22Ch32Ord22Level1(testCase)
            
            dec = 2;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch32Ord22Level2eriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 3 2 ];
            ch = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
            
        end
        
        % Test
        function testStepDec22Ch43Ord22Level1(testCase)
            
            dec = 2;
            decch = [ dec dec 4 3 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch43Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 4 3 ];
            ch = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv1{2};
            coefs{9} = coefsExpctdLv1{3};
            coefs{10} = coefsExpctdLv1{4};
            coefs{11} = coefsExpctdLv1{5};
            coefs{12} = coefsExpctdLv1{6};
            coefs{13} = coefsExpctdLv1{7};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
            
        end
 
        % Test
        function testStepDec22Ch9Ord22Level1(testCase)
            
            dec = 2;
            decch = [ dec dec 5 4 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels', decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
            
        end
        
        % Test
        function testSteppDec22Ch9Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 5 4 ];
            decch = [ dec dec nChs ];
            ch = sum(nChs);
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv2{8};
            coefs{9} = coefsExpctdLv2{9};
            coefs{10} = coefsExpctdLv1{2};
            coefs{11} = coefsExpctdLv1{3};
            coefs{12} = coefsExpctdLv1{4};
            coefs{13} = coefsExpctdLv1{5};
            coefs{14} = coefsExpctdLv1{6};
            coefs{15} = coefsExpctdLv1{7};
            coefs{16} = coefsExpctdLv1{8};
            coefs{17} = coefsExpctdLv1{9};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
            
        end
        
        % Level 3, dec 11 ch 54 order 88
        function testStepDec11Ch54Ord88Level3PeriodicExt(testCase)
            
            dec = 1;
            nChs = [ 5 4 ];
            decch = [ dec dec nChs ];
            ch = sum(nChs);
            ord = 8;
            height = 64;
            width = 64;
            srcImg = rand(height,width);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv3{5};
            coefs{6} = coefsExpctdLv3{6};
            coefs{7} = coefsExpctdLv3{7};
            coefs{8} = coefsExpctdLv3{8};
            coefs{9} = coefsExpctdLv3{9};
            coefs{10} = coefsExpctdLv2{2};
            coefs{11} = coefsExpctdLv2{3};
            coefs{12} = coefsExpctdLv2{4};
            coefs{13} = coefsExpctdLv2{5};
            coefs{14} = coefsExpctdLv2{6};
            coefs{15} = coefsExpctdLv2{7};
            coefs{16} = coefsExpctdLv2{8};
            coefs{17} = coefsExpctdLv2{9};
            coefs{18} = coefsExpctdLv1{2};
            coefs{19} = coefsExpctdLv1{3};
            coefs{20} = coefsExpctdLv1{4};
            coefs{21} = coefsExpctdLv1{5};
            coefs{22} = coefsExpctdLv1{6};
            coefs{23} = coefsExpctdLv1{7};
            coefs{24} = coefsExpctdLv1{8};
            coefs{25} = coefsExpctdLv1{9};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Level 3, dec 22 ch 54 order 44
        function testStepDec22Ch54Ord44Level3PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 5 4 ];
            decch = [ dec dec nChs ];
            ch = sum(nChs);
            ord = 4;
            height = 64;
            width = 64;
            srcImg = rand(height,width);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv3{5};
            coefs{6} = coefsExpctdLv3{6};
            coefs{7} = coefsExpctdLv3{7};
            coefs{8} = coefsExpctdLv3{8};
            coefs{9} = coefsExpctdLv3{9};
            coefs{10} = coefsExpctdLv2{2};
            coefs{11} = coefsExpctdLv2{3};
            coefs{12} = coefsExpctdLv2{4};
            coefs{13} = coefsExpctdLv2{5};
            coefs{14} = coefsExpctdLv2{6};
            coefs{15} = coefsExpctdLv2{7};
            coefs{16} = coefsExpctdLv2{8};
            coefs{17} = coefsExpctdLv2{9};
            coefs{18} = coefsExpctdLv1{2};
            coefs{19} = coefsExpctdLv1{3};
            coefs{20} = coefsExpctdLv1{4};
            coefs{21} = coefsExpctdLv1{5};
            coefs{22} = coefsExpctdLv1{6};
            coefs{23} = coefsExpctdLv1{7};
            coefs{24} = coefsExpctdLv1{8};
            coefs{25} = coefsExpctdLv1{9};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch32Ord22Level1PeriodicExt(testCase)
            
            decch = [2 2 3 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
            
        end
        
        % Test
        function testStepDec22Ch32Ord22Level2PeriodicExt(testCase)
            
            decch = [2 2 3 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(1)).',decch(2));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(1)).',decch(2));
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch42Ord22Level1PeriodicExt(testCase)
            
            decch = [2 2 4 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch42Ord22Level2PeriodicExt(testCase)
            
            decch = [2 2 4 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(1)).',decch(2));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(1)).',decch(2));
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv1{2};
            coefs{8} = coefsExpctdLv1{3};
            coefs{9} = coefsExpctdLv1{4};
            coefs{10} = coefsExpctdLv1{5};
            coefs{11} = coefsExpctdLv1{6};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testSetLpPuFb2dDec22Ch52Ord44(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 5 2 ];
            ord = [ 4 4 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcImg,nLevels);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end
        
        % Test
        function testIsCloneFalse(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 6 2 ];
            ord = [ 4 4 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination',...
                'IsCloneLpPuFb2d',true);
            
            % Pre
            coefsPre1 = step(testCase.analyzer,srcImg,nLevels);
            
            % Pst
            angs = randn(size(get(lppufb,'Angles')));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre1(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,sprintf('%g',diff));
            
            % Instantiation of target class
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination',...
                'IsCloneLpPuFb2d',false);
            
            % Pre
            coefsPre1 = step(testCase.analyzer,srcImg,nLevels);
            
            % Pst
            angs = randn(size(get(lppufb,'Angles')));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst1(:)-coefsPre1(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));            
           
        end
        
        % Test
        function testClone(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 5 3 ];
            ord = [ 4 4 ];
            height = 64;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.analyzer = TypeIIAnalysisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination');
            %s = matlab.System.saveObject(testCase.analyzer);

            % Clone
            cloneAnalyzer = clone(testCase.analyzer);
            
            % Evaluation
            testCase.verifyEqual(cloneAnalyzer,testCase.analyzer);
            testCase.verifyFalse(cloneAnalyzer == testCase.analyzer);
            prpOrg = get(testCase.analyzer,'LpPuFb2d');
            prpCln = get(cloneAnalyzer,'LpPuFb2d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %            
            [coefExpctd,scaleExpctd] = step(testCase.analyzer,srcImg,nLevels);
            [coefActual,scaleActual] = step(cloneAnalyzer,srcImg,nLevels);
            testCase.assertEqual(coefActual,coefExpctd);
            testCase.assertEqual(scaleActual,scaleExpctd);
            
        end
        
    end
    
end

