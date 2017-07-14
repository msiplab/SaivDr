classdef ComplexAnalysis3dSystemTestCase < matlab.unittest.TestCase
    %ComplexAnalysis3dSystemTESTCASE Test case for ComplexAnalysis3dSystem
    %
    % SVN identifier:
    % $Id: ComplexAnalysis3dSystemTestCase.m 866 2015-11-24 04:29:42Z sho $
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2015, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627    
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
            import saivdr.dictionary.generalfb.*
            analysisFiltersExpctd = [];
            decimationFactorExpctd =  [ 2 2 2 ];
            filterDomainExpctd = 'Spatial';
            boundaryOperationExpctd = 'Circular';
            
            % Instantiation
            testCase.analyzer = ComplexAnalysis3dSystem();
            
            % Actual value
            analysisFiltersActual = get(testCase.analyzer,'AnalysisFilters');
            decimationFactorActual = get(testCase.analyzer,'DecimationFactor');
            filterDomainActual = get(testCase.analyzer,'FilterDomain');
            boundaryOperationActual = get(testCase.analyzer,'BoundaryOperation');            
            
            % Evaluation
            testCase.assertEqual(analysisFiltersActual,analysisFiltersExpctd);
            testCase.assertEqual(decimationFactorActual,decimationFactorExpctd);
            testCase.assertEqual(filterDomainActual,filterDomainExpctd);
            testCase.assertEqual(boundaryOperationActual,boundaryOperationExpctd);  
            
        end
        
        % Test
        function testAnalysisFilters(testCase)
            
            % Expected values
            analysisFiltersExpctd(:,:,:,1) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFiltersExpctd(:,:,:,2) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFiltersExpctd(:,:,:,3) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFiltersExpctd(:,:,:,4) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFiltersExpctd(:,:,:,5) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFiltersExpctd(:,:,:,6) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFiltersExpctd(:,:,:,7) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFiltersExpctd(:,:,:,8) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
                        
            % Instantiation
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'AnalysisFilters',analysisFiltersExpctd);
            
            % Actual value
            analysisFiltersActual = get(testCase.analyzer,'AnalysisFilters');
            
            % Evaluation
            nChs = size(analysisFiltersExpctd,4);
            for iCh = 1:nChs
                testCase.assertEqual(analysisFiltersActual(:,:,:,iCh),...
                    analysisFiltersExpctd(:,:,:,iCh));
            end
            
        end
        
        % Test
        function testStepDec222Ch44Ord000Level1(testCase)

            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 2 ]; 
            analysisFilters(:,:,:,1) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,2) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,3) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,4) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,5) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,6) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,7) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,8) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,4);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end

        % Test
        function testStepDec222Ch54Ord000Level1(testCase)

            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 2 ]; 
            analysisFilters(:,:,:,1) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,2) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,3) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,4) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,5) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,6) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,7) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,8) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,9) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,4);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);                
                subCoef = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end

        % Test        
        function testStepDec222Ch54Ord222Level1(testCase)

            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 2 ]; 
            analysisFilters(:,:,:,1) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,2) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,3) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,4) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,5) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,6) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,7) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,8) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,9) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,4);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end

        % Test
        function testStepDec111Ch54Ord111Level1(testCase)

            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 1 1 1 ]; 
            analysisFilters(:,:,:,1) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,2) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,3) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,4) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,5) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,6) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,7) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,8) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,9) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,4);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end
        
        % Test        
        function testStepDec321Ch44Ord222Level1(testCase)

            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 3 2 1 ]; 
            analysisFilters(:,:,:,1) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,2) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,3) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,4) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,5) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,6) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,7) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,8) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,4);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end
        
        % Test
        function testStepDec222Ch44Ord222Level2(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 2 ]; 
            analysisFilters(:,:,:,1) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,2) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,3) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,4) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,5) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,6) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,7) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,8) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));        
            nLevels = 2;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);              
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};        
            coefs{7} = coefsExpctdLv2{7};                
            coefs{8} = coefsExpctdLv2{8};                        
            coefs{9} = coefsExpctdLv1{2};
            coefs{10} = coefsExpctdLv1{3};            
            coefs{11} = coefsExpctdLv1{4};
            coefs{12} = coefsExpctdLv1{5};
            coefs{13} = coefsExpctdLv1{6};                    
            coefs{14} = coefsExpctdLv1{7};
            coefs{15} = coefsExpctdLv1{8};                            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));  
            
        end
        
        % Test
        function testStepDec222Ch44Ord222Level3(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 2 ];
            analysisFilters(:,:,:,1) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,2) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,3) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,4) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,5) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,6) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,7) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,8) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            nLevels = 3;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv3{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv2{1},h,...
                    'conv','circ'),...
                    nDecs);
            end            
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv3{5};
            coefs{6} = coefsExpctdLv3{6};
            coefs{7} = coefsExpctdLv3{7};
            coefs{8} = coefsExpctdLv3{8};
            coefs{9} = coefsExpctdLv2{2};
            coefs{10} = coefsExpctdLv2{3};
            coefs{11} = coefsExpctdLv2{4};
            coefs{12} = coefsExpctdLv2{5};
            coefs{13} = coefsExpctdLv2{6};
            coefs{14} = coefsExpctdLv2{7};
            coefs{15} = coefsExpctdLv2{8};            
            coefs{16} = coefsExpctdLv1{2};
            coefs{17} = coefsExpctdLv1{3};
            coefs{18} = coefsExpctdLv1{4};
            coefs{19} = coefsExpctdLv1{5};
            coefs{20} = coefsExpctdLv1{6};
            coefs{21} = coefsExpctdLv1{7};
            coefs{22} = coefsExpctdLv1{8};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec222Ch55Ord444Level3(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 2 ];
            analysisFilters(:,:,:,1) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,2) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,3) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,4) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,5) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,6) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,7) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,8) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,9) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,10) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            nLevels = 3;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv3{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv2{1},h,...
                    'conv','circ'),...
                    nDecs);
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
            coefs{10} = coefsExpctdLv3{10};            
            coefs{11} = coefsExpctdLv2{2};
            coefs{12} = coefsExpctdLv2{3};
            coefs{13} = coefsExpctdLv2{4};            
            coefs{14} = coefsExpctdLv2{5};
            coefs{15} = coefsExpctdLv2{6};
            coefs{16} = coefsExpctdLv2{7};            
            coefs{17} = coefsExpctdLv2{8};                        
            coefs{18} = coefsExpctdLv2{9};            
            coefs{19} = coefsExpctdLv2{10};                                    
            coefs{20} = coefsExpctdLv1{2};
            coefs{21} = coefsExpctdLv1{3};
            coefs{22} = coefsExpctdLv1{4};
            coefs{23} = coefsExpctdLv1{5};
            coefs{24} = coefsExpctdLv1{6};
            coefs{25} = coefsExpctdLv1{7};
            coefs{26} = coefsExpctdLv1{8};
            coefs{27} = coefsExpctdLv1{9};
            coefs{28} = coefsExpctdLv1{10};            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));            
            
        end
        
                % Test
        function testStepDec222Ch44Ord000Level1Freq(testCase)

            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 2 ]; 
            analysisFilters(:,:,:,1) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,2) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,3) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,4) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,5) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,6) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,7) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,8) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,4);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end

        % Test
        function testStepDec222Ch54Ord000Level1Freq(testCase)

            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 2 ]; 
            analysisFilters(:,:,:,1) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,2) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,3) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,4) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,5) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,6) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,7) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,8) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,9) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,4);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);                
                subCoef = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end

        % Test        
        function testStepDec222Ch54Ord222Level1Freq(testCase)

            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 2 ]; 
            analysisFilters(:,:,:,1) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,2) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,3) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,4) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,5) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,6) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,7) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,8) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,9) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,4);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end

        % Test
        function testStepDec111Ch54Ord111Level1Freq(testCase)

            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 1 1 1 ]; 
            analysisFilters(:,:,:,1) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,2) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,3) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,4) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,5) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,6) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,7) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,8) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            analysisFilters(:,:,:,9) = randn(2,2,2).*exp(1i*2*pi*rand(2,2,2));
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,4);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end
        
        % Test        
        function testStepDec321Ch44Ord222Level1Freq(testCase)

            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 3 2 1 ]; 
            analysisFilters(:,:,:,1) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,2) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,3) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,4) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,5) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,6) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,7) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,8) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,4);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end
        
        % Test
        function testStepDec222Ch44Ord222Level2Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 2 ]; 
            analysisFilters(:,:,:,1) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,2) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,3) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,4) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,5) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,6) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,7) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,8) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));        
            nLevels = 2;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);              
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};        
            coefs{7} = coefsExpctdLv2{7};                
            coefs{8} = coefsExpctdLv2{8};                        
            coefs{9} = coefsExpctdLv1{2};
            coefs{10} = coefsExpctdLv1{3};            
            coefs{11} = coefsExpctdLv1{4};
            coefs{12} = coefsExpctdLv1{5};
            coefs{13} = coefsExpctdLv1{6};                    
            coefs{14} = coefsExpctdLv1{7};
            coefs{15} = coefsExpctdLv1{8};                            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));  
            
        end
        
        % Test
        function testStepDec222Ch44Ord222Level3Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 2 ];
            analysisFilters(:,:,:,1) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,2) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,3) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,4) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,5) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,6) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,7) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            analysisFilters(:,:,:,8) = randn(6,6,6).*exp(1i*2*pi*rand(6,6,6));
            nLevels = 3;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv3{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv2{1},h,...
                    'conv','circ'),...
                    nDecs);
            end            
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv3{5};
            coefs{6} = coefsExpctdLv3{6};
            coefs{7} = coefsExpctdLv3{7};
            coefs{8} = coefsExpctdLv3{8};
            coefs{9} = coefsExpctdLv2{2};
            coefs{10} = coefsExpctdLv2{3};
            coefs{11} = coefsExpctdLv2{4};
            coefs{12} = coefsExpctdLv2{5};
            coefs{13} = coefsExpctdLv2{6};
            coefs{14} = coefsExpctdLv2{7};
            coefs{15} = coefsExpctdLv2{8};            
            coefs{16} = coefsExpctdLv1{2};
            coefs{17} = coefsExpctdLv1{3};
            coefs{18} = coefsExpctdLv1{4};
            coefs{19} = coefsExpctdLv1{5};
            coefs{20} = coefsExpctdLv1{6};
            coefs{21} = coefsExpctdLv1{7};
            coefs{22} = coefsExpctdLv1{8};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec222Ch55Ord444Level3Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 48;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 2 ];
            analysisFilters(:,:,:,1) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,2) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,3) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,4) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,5) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,6) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,7) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,8) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,9) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            analysisFilters(:,:,:,10) = randn(10,10,10).*exp(1i*2*pi*rand(10,10,10));
            nLevels = 3;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv3{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv2{1},h,...
                    'conv','circ'),...
                    nDecs);
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
            coefs{10} = coefsExpctdLv3{10};            
            coefs{11} = coefsExpctdLv2{2};
            coefs{12} = coefsExpctdLv2{3};
            coefs{13} = coefsExpctdLv2{4};            
            coefs{14} = coefsExpctdLv2{5};
            coefs{15} = coefsExpctdLv2{6};
            coefs{16} = coefsExpctdLv2{7};            
            coefs{17} = coefsExpctdLv2{8};                        
            coefs{18} = coefsExpctdLv2{9};            
            coefs{19} = coefsExpctdLv2{10};                                    
            coefs{20} = coefsExpctdLv1{2};
            coefs{21} = coefsExpctdLv1{3};
            coefs{22} = coefsExpctdLv1{4};
            coefs{23} = coefsExpctdLv1{5};
            coefs{24} = coefsExpctdLv1{6};
            coefs{25} = coefsExpctdLv1{7};
            coefs{26} = coefsExpctdLv1{8};
            coefs{27} = coefsExpctdLv1{9};
            coefs{28} = coefsExpctdLv1{10};            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));            
            
        end
        
        % Test
        function testStepDec321Ch44Ord222Level2(testCase) 
            
            % Parameters
            height = 108;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 3 2 1 ];
            analysisFilters(:,:,:,1) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,2) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,3) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,4) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,5) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,6) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,7) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,8) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            nLevels = 2;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv2{8};
            coefs{9} = coefsExpctdLv1{2};
            coefs{10} = coefsExpctdLv1{3};
            coefs{11} = coefsExpctdLv1{4};
            coefs{12} = coefsExpctdLv1{5};
            coefs{13} = coefsExpctdLv1{6};
            coefs{14} = coefsExpctdLv1{7};
            coefs{15} = coefsExpctdLv1{8};            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec321Ch44Ord222Level2Freq(testCase) 
            
            % Parameters
            height = 108;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 3 2 1 ];
            analysisFilters(:,:,:,1) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,2) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,3) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,4) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,5) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,6) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,7) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,8) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            nLevels = 2;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv2{8};
            coefs{9} = coefsExpctdLv1{2};
            coefs{10} = coefsExpctdLv1{3};
            coefs{11} = coefsExpctdLv1{4};
            coefs{12} = coefsExpctdLv1{5};
            coefs{13} = coefsExpctdLv1{6};
            coefs{14} = coefsExpctdLv1{7};
            coefs{15} = coefsExpctdLv1{8};            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec321Ch44Ord222Level3(testCase) 
            
            % Parameters
            height = 108;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 3 2 1 ];
            analysisFilters(:,:,:,1) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,2) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,3) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,4) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,5) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,6) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,7) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,8) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            nLevels = 3;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv3{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv2{1},h,...
                    'conv','circ'),...
                    nDecs);
            end            
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv3{5};
            coefs{6} = coefsExpctdLv3{6};
            coefs{7} = coefsExpctdLv3{7};
            coefs{8} = coefsExpctdLv3{8};
            coefs{9} = coefsExpctdLv2{2};
            coefs{10} = coefsExpctdLv2{3};
            coefs{11} = coefsExpctdLv2{4};
            coefs{12} = coefsExpctdLv2{5};
            coefs{13} = coefsExpctdLv2{6};
            coefs{14} = coefsExpctdLv2{7};
            coefs{15} = coefsExpctdLv2{8};            
            coefs{16} = coefsExpctdLv1{2};
            coefs{17} = coefsExpctdLv1{3};
            coefs{18} = coefsExpctdLv1{4};
            coefs{19} = coefsExpctdLv1{5};
            coefs{20} = coefsExpctdLv1{6};
            coefs{21} = coefsExpctdLv1{7};
            coefs{22} = coefsExpctdLv1{8};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec321Ch44Ord222Level3Freq(testCase) 
            
            % Parameters
            height = 108;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 3 2 1 ];
            analysisFilters(:,:,:,1) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,2) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,3) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,4) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,5) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,6) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,7) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,8) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            nLevels = 3;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv3{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv2{1},h,...
                    'conv','circ'),...
                    nDecs);
            end            
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv3{5};
            coefs{6} = coefsExpctdLv3{6};
            coefs{7} = coefsExpctdLv3{7};
            coefs{8} = coefsExpctdLv3{8};
            coefs{9} = coefsExpctdLv2{2};
            coefs{10} = coefsExpctdLv2{3};
            coefs{11} = coefsExpctdLv2{4};
            coefs{12} = coefsExpctdLv2{5};
            coefs{13} = coefsExpctdLv2{6};
            coefs{14} = coefsExpctdLv2{7};
            coefs{15} = coefsExpctdLv2{8};            
            coefs{16} = coefsExpctdLv1{2};
            coefs{17} = coefsExpctdLv1{3};
            coefs{18} = coefsExpctdLv1{4};
            coefs{19} = coefsExpctdLv1{5};
            coefs{20} = coefsExpctdLv1{6};
            coefs{21} = coefsExpctdLv1{7};
            coefs{22} = coefsExpctdLv1{8};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec234Ch1414Ord222Level2(testCase) 
            
            % Parameters
            height = 8*2^2;
            width = 12*3^2;
            depth = 16*4^2;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 3 4 ];
            analysisFilters = zeros(6,9,12,28);
            for iCh = 1:28
                analysisFilters(:,:,:,iCh) = randn(6,9,12).*exp(1i*2*pi*rand(6,9,12));
            end
            nLevels = 2;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefs = cell(nLevels*(nChs-1)+1,1);
            coefs{1} = coefsExpctdLv2{1};            
            for iCh = 2:nChs
               coefs{iCh} = coefsExpctdLv2{iCh};
               coefs{iCh+27} = coefsExpctdLv1{iCh};
            end
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Spatial');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        
        % Test
        function testStepDec234Ch1414Ord222Level2Freq(testCase) 
            
            % Parameters
            height = 8*2^2;
            width = 12*3^2;
            depth = 16*4^2;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 3 4 ];
            analysisFilters = zeros(6,9,12,28);
            for iCh = 1:28
                analysisFilters(:,:,:,iCh) = randn(6,9,12).*exp(1i*2*pi*rand(6,9,12));
            end
            nLevels = 2;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefs = cell(nLevels*(nChs-1)+1,1);
            coefs{1} = coefsExpctdLv2{1};            
            for iCh = 2:nChs
               coefs{iCh} = coefsExpctdLv2{iCh};
               coefs{iCh+27} = coefsExpctdLv1{iCh};
            end
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        
        % Test
        function testClone(testCase) 
            
            % Parameters
            height = 108;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 3 2 1 ];
            analysisFilters(:,:,:,1) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,2) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,3) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,4) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,5) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,6) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,7) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            analysisFilters(:,:,:,8) = randn(9,6,3).*exp(1i*2*pi*rand(9,6,3));
            nLevels = 3;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,4);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},h,...
                    'conv','circ'),...
                    nDecs);
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                h = analysisFilters(:,:,:,iSubband);
                coefsExpctdLv3{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv2{1},h,...
                    'conv','circ'),...
                    nDecs);
            end            
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv3{5};
            coefs{6} = coefsExpctdLv3{6};
            coefs{7} = coefsExpctdLv3{7};
            coefs{8} = coefsExpctdLv3{8};
            coefs{9} = coefsExpctdLv2{2};
            coefs{10} = coefsExpctdLv2{3};
            coefs{11} = coefsExpctdLv2{4};
            coefs{12} = coefsExpctdLv2{5};
            coefs{13} = coefsExpctdLv2{6};
            coefs{14} = coefsExpctdLv2{7};
            coefs{15} = coefsExpctdLv2{8};            
            coefs{16} = coefsExpctdLv1{2};
            coefs{17} = coefsExpctdLv1{3};
            coefs{18} = coefsExpctdLv1{4};
            coefs{19} = coefsExpctdLv1{5};
            coefs{20} = coefsExpctdLv1{6};
            coefs{21} = coefsExpctdLv1{7};
            coefs{22} = coefsExpctdLv1{8};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = ComplexAnalysis3dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency');
            cloneAnalyzer = clone(testCase.analyzer);
            
            % Actual values
            [coefsActual,scalesActual] = step(cloneAnalyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end       
    end
    
end