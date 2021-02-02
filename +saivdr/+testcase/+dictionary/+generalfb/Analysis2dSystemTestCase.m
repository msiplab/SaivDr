classdef Analysis2dSystemTestCase < matlab.unittest.TestCase
    %ANALYSIS2DSYSTEMTESTCASE Test case for Analysis2dSystem
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015-2020, Shogo MURAMATSU
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
            import saivdr.dictionary.generalfb.*
            analysisFiltersExpctd = [];
            decimationFactorExpctd =  [ 2 2 ];
            filterDomainExpctd = 'Spatial';
            boundaryOperationExpctd = 'Circular';
            
            % Instantiation
            testCase.analyzer = Analysis2dSystem();
            
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
            analysisFiltersExpctd(:,:,1) = randn(2,2);
            analysisFiltersExpctd(:,:,2) = randn(2,2);
            analysisFiltersExpctd(:,:,3) = randn(2,2);
            analysisFiltersExpctd(:,:,4) = randn(2,2);
                        
            % Instantiation
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'AnalysisFilters',analysisFiltersExpctd);
            
            % Actual value
            analysisFiltersActual = get(testCase.analyzer,'AnalysisFilters');
            
            % Evaluation
            nChs = size(analysisFiltersExpctd,3);
            for iCh = 1:nChs
                testCase.assertEqual(analysisFiltersActual(:,:,iCh),...
                    analysisFiltersExpctd(:,:,iCh));
            end
            
        end        

        % Test
        function testStepDec22Ch22Ord00Level1(testCase)

            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 2 2 ]; 
            analysisFilters(:,:,1) = randn(2,2);
            analysisFilters(:,:,2) = randn(2,2);
            analysisFilters(:,:,3) = randn(2,2);
            analysisFilters(:,:,4) = randn(2,2);
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));

        end

        % Test
        function testStepDec22Ch33Ord00Level1(testCase)

            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 2 2 ]; 
            analysisFilters(:,:,1) = randn(2,2);
            analysisFilters(:,:,2) = randn(2,2);
            analysisFilters(:,:,3) = randn(2,2);
            analysisFilters(:,:,4) = randn(2,2);
            analysisFilters(:,:,5) = randn(2,2);
            analysisFilters(:,:,6) = randn(2,2);            
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));

        end

        % Test        
        function testStepDec22Ch33Ord22Level1(testCase)

            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 2 2 ]; 
            analysisFilters(:,:,1) = randn(6,6);
            analysisFilters(:,:,2) = randn(6,6);
            analysisFilters(:,:,3) = randn(6,6);
            analysisFilters(:,:,4) = randn(6,6);
            analysisFilters(:,:,5) = randn(6,6);
            analysisFilters(:,:,6) = randn(6,6);            
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));

        end

        % Test
        function testStepDec11Ch33Ord11Level1(testCase)

            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 1 1 ]; 
            analysisFilters(:,:,1) = randn(2,2);
            analysisFilters(:,:,2) = randn(2,2);
            analysisFilters(:,:,3) = randn(2,2);
            analysisFilters(:,:,4) = randn(2,2);
            analysisFilters(:,:,5) = randn(2,2);
            analysisFilters(:,:,6) = randn(2,2);            
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));

        end
        
        % Test        
        function testStepDec33Ch54Ord22Level1(testCase)

            % Parameters
            height = 48;
            width = 63;
            srcImg = rand(height,width);
            nDecs = [ 3 3 ]; 
            analysisFilters(:,:,1) = randn(9,9);
            analysisFilters(:,:,2) = randn(9,9);
            analysisFilters(:,:,3) = randn(9,9);
            analysisFilters(:,:,4) = randn(9,9);
            analysisFilters(:,:,5) = randn(9,9);
            analysisFilters(:,:,6) = randn(9,9);
            analysisFilters(:,:,7) = randn(9,9);
            analysisFilters(:,:,8) = randn(9,9);
            analysisFilters(:,:,9) = randn(9,9);            
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));

        end
        
        % Test        
        function testStepDec44Ch1212Ord22Level1(testCase)

            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 4 4 ]; 
            analysisFilters = zeros(12,12,24);
            for iCh = 1:24
                analysisFilters(:,:,iCh) = randn(12,12);
            end
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));

        end
               
        % Test
        function testStepDec22Ch22Ord22Level2(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 2 2 ]; 
            analysisFilters(:,:,1) = randn(6,6);
            analysisFilters(:,:,2) = randn(6,6);
            analysisFilters(:,:,3) = randn(6,6);
            analysisFilters(:,:,4) = randn(6,6);
            nLevels = 2;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};
            coefs{7} = coefsExpctdLv1{4};            
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
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));  
            
        end
        
        % Test
        function testStepDec22Ch22Ord22Level3(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 2 2 ];
            analysisFilters(:,:,1) = randn(6,6);
            analysisFilters(:,:,2) = randn(6,6);
            analysisFilters(:,:,3) = randn(6,6);
            analysisFilters(:,:,4) = randn(6,6);
            nLevels = 3;
            
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end            
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv2{2};
            coefs{6} = coefsExpctdLv2{3};
            coefs{7} = coefsExpctdLv2{4};            
            coefs{8} = coefsExpctdLv1{2};
            coefs{9} = coefsExpctdLv1{3};
            coefs{10} = coefsExpctdLv1{4};
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
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch44Ord44Level3(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 2 2 ];
            analysisFilters(:,:,1) = randn(10,10);
            analysisFilters(:,:,2) = randn(10,10);
            analysisFilters(:,:,3) = randn(10,10);
            analysisFilters(:,:,4) = randn(10,10);
            analysisFilters(:,:,5) = randn(10,10);
            analysisFilters(:,:,6) = randn(10,10);
            analysisFilters(:,:,7) = randn(10,10);
            analysisFilters(:,:,8) = randn(10,10);
            nLevels = 3;
            
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
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
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            
        end
        
        % Test
        function testStepDec22Ch22Ord00Level1Freq(testCase)

            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 2 2 ]; 
            analysisFilters(:,:,1) = randn(2,2);
            analysisFilters(:,:,2) = randn(2,2);
            analysisFilters(:,:,3) = randn(2,2);
            analysisFilters(:,:,4) = randn(2,2);
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));

        end

        % Test
        function testStepDec22Ch33Ord00Level1Freq(testCase)

            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 2 2 ]; 
            analysisFilters(:,:,1) = randn(2,2);
            analysisFilters(:,:,2) = randn(2,2);
            analysisFilters(:,:,3) = randn(2,2);
            analysisFilters(:,:,4) = randn(2,2);
            analysisFilters(:,:,5) = randn(2,2);
            analysisFilters(:,:,6) = randn(2,2);            
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));

        end

        % Test        
        function testStepDec22Ch33Ord22Level1Freq(testCase)

            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 2 2 ]; 
            analysisFilters(:,:,1) = randn(6,6);
            analysisFilters(:,:,2) = randn(6,6);
            analysisFilters(:,:,3) = randn(6,6);
            analysisFilters(:,:,4) = randn(6,6);
            analysisFilters(:,:,5) = randn(6,6);
            analysisFilters(:,:,6) = randn(6,6);            
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);
            
            % Instantiation of target class
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec11Ch33Ord11Level1Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 1 1 ];
            analysisFilters(:,:,1) = randn(2,2);
            analysisFilters(:,:,2) = randn(2,2);
            analysisFilters(:,:,3) = randn(2,2);
            analysisFilters(:,:,4) = randn(2,2);
            analysisFilters(:,:,5) = randn(2,2);
            analysisFilters(:,:,6) = randn(2,2);
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);
            
            % Instantiation of target class
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);                
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec33Ch54Ord22Level1Freq(testCase)
            
            % Parameters
            height = 48;
            width = 63;
            srcImg = rand(height,width);
            nDecs = [ 3 3 ];
            analysisFilters(:,:,1) = randn(9,9);
            analysisFilters(:,:,2) = randn(9,9);
            analysisFilters(:,:,3) = randn(9,9);
            analysisFilters(:,:,4) = randn(9,9);
            analysisFilters(:,:,5) = randn(9,9);
            analysisFilters(:,:,6) = randn(9,9);
            analysisFilters(:,:,7) = randn(9,9);
            analysisFilters(:,:,8) = randn(9,9);
            analysisFilters(:,:,9) = randn(9,9);
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);
            
            % Instantiation of target class
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);                
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
        
        
        % Test
        function testStepDec44Ch1212Ord22Level1Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 4 4 ];
            analysisFilters = zeros(12,12,24);
            for iCh = 1:24
                analysisFilters(:,:,iCh) = randn(12,12);
            end
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);
            
            % Instantiation of target class
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);                
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch22Ord22Level2Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 2 2 ]; 
            analysisFilters(:,:,1) = randn(6,6);
            analysisFilters(:,:,2) = randn(6,6);
            analysisFilters(:,:,3) = randn(6,6);
            analysisFilters(:,:,4) = randn(6,6);
            nLevels = 2;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};
            coefs{7} = coefsExpctdLv1{4};            
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
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                 'FilterDomain','Frequency',...
                 'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));  
            
        end
        
        % Test
        function testStepDec22Ch22Ord22Level3Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 2 2 ];
            analysisFilters(:,:,1) = randn(6,6);
            analysisFilters(:,:,2) = randn(6,6);
            analysisFilters(:,:,3) = randn(6,6);
            analysisFilters(:,:,4) = randn(6,6);
            nLevels = 3;
            
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end            
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv2{2};
            coefs{6} = coefsExpctdLv2{3};
            coefs{7} = coefsExpctdLv2{4};            
            coefs{8} = coefsExpctdLv1{2};
            coefs{9} = coefsExpctdLv1{3};
            coefs{10} = coefsExpctdLv1{4};
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
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                 'FilterDomain','Frequency',...
                 'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch44Ord44Level3Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            srcImg = rand(height,width);
            nDecs = [ 2 2 ];
            analysisFilters(:,:,1) = randn(10,10);
            analysisFilters(:,:,2) = randn(10,10);
            analysisFilters(:,:,3) = randn(10,10);
            analysisFilters(:,:,4) = randn(10,10);
            analysisFilters(:,:,5) = randn(10,10);
            analysisFilters(:,:,6) = randn(10,10);
            analysisFilters(:,:,7) = randn(10,10);
            analysisFilters(:,:,8) = randn(10,10);
            nLevels = 3;
            
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
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
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            
        end
        
        % Test
        function testStepDec33Ch54Ord22Level3(testCase)
            
            % Parameters
            height = 108;
            width = 135;
            srcImg = rand(height,width);
            nDecs = [ 3 3 ];
            analysisFilters(:,:,1) = randn(9,9);
            analysisFilters(:,:,2) = randn(9,9);
            analysisFilters(:,:,3) = randn(9,9);
            analysisFilters(:,:,4) = randn(9,9);
            analysisFilters(:,:,5) = randn(9,9);
            analysisFilters(:,:,6) = randn(9,9);
            analysisFilters(:,:,7) = randn(9,9);
            analysisFilters(:,:,8) = randn(9,9);
            analysisFilters(:,:,9) = randn(9,9);
            nLevels = 3;
            
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
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
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            
        end
        
        % Test
        function testStepDec44Ch1212Ord22Level3(testCase)
            
            % Parameters
            height = 256;
            width = 320;
            srcImg = rand(height,width);
            nDecs = [ 4 4 ];
            analysisFilters = zeros(12,12,24);
            for iCh = 1:24
                analysisFilters(:,:,iCh) = randn(12,12);
            end
            nLevels = 3;
            
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end            
            coefs = cell(70,1);
            coefs{1} = coefsExpctdLv3{1};
            for iCh = 2:24
                coefs{iCh}    = coefsExpctdLv3{iCh};
                coefs{iCh+23} = coefsExpctdLv2{iCh};
                coefs{iCh+46} = coefsExpctdLv1{iCh};                
            end
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
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            
        end
        
        % Test
        function testStepDec33Ch54Ord22Level3Freq(testCase)
            
            % Parameters
            height = 108;
            width = 135;
            srcImg = rand(height,width);
            nDecs = [ 3 3 ];
            analysisFilters(:,:,1) = randn(9,9);
            analysisFilters(:,:,2) = randn(9,9);
            analysisFilters(:,:,3) = randn(9,9);
            analysisFilters(:,:,4) = randn(9,9);
            analysisFilters(:,:,5) = randn(9,9);
            analysisFilters(:,:,6) = randn(9,9);
            analysisFilters(:,:,7) = randn(9,9);
            analysisFilters(:,:,8) = randn(9,9);
            analysisFilters(:,:,9) = randn(9,9);
            nLevels = 3;
            
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
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
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            
        end
        
        % Test
        function testStepDec44Ch1212Ord22Level3Freq(testCase)
            
            % Parameters
            height = 256;
            width = 320;
            srcImg = rand(height,width);
            nDecs = [ 4 4 ];
            analysisFilters = zeros(12,12,24);
            for iCh = 1:24
                analysisFilters(:,:,iCh) = randn(12,12);
            end
            nLevels = 3;
            
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv3 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end            
            coefs = cell(70,1);
            coefs{1} = coefsExpctdLv3{1};
            for iCh = 2:24
                coefs{iCh}    = coefsExpctdLv3{iCh};
                coefs{iCh+23} = coefsExpctdLv2{iCh};
                coefs{iCh+46} = coefsExpctdLv1{iCh};                
            end
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
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-9,...
                sprintf('%g',diff));            
            
        end

        % Test
        function testStepDec33Ch54Ord22Level2(testCase)
            
            % Parameters
            height = 108;
            width = 135;
            srcImg = rand(height,width);
            nDecs = [ 3 3 ];
            analysisFilters(:,:,1) = randn(9,9);
            analysisFilters(:,:,2) = randn(9,9);
            analysisFilters(:,:,3) = randn(9,9);
            analysisFilters(:,:,4) = randn(9,9);
            analysisFilters(:,:,5) = randn(9,9);
            analysisFilters(:,:,6) = randn(9,9);
            analysisFilters(:,:,7) = randn(9,9);
            analysisFilters(:,:,8) = randn(9,9);
            analysisFilters(:,:,9) = randn(9,9);
            nLevels = 2;
            
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
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
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            
        end
                
        % Test
        function testStepDec33Ch54Ord22Level2Freq(testCase)
            
            % Parameters
            height = 108;
            width = 135;
            srcImg = rand(height,width);
            nDecs = [ 3 3 ];
            analysisFilters(:,:,1) = randn(9,9);
            analysisFilters(:,:,2) = randn(9,9);
            analysisFilters(:,:,3) = randn(9,9);
            analysisFilters(:,:,4) = randn(9,9);
            analysisFilters(:,:,5) = randn(9,9);
            analysisFilters(:,:,6) = randn(9,9);
            analysisFilters(:,:,7) = randn(9,9);
            analysisFilters(:,:,8) = randn(9,9);
            analysisFilters(:,:,9) = randn(9,9);
            nLevels = 2;
            
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
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
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            
        end        
        
               % Test
        function testStepDec33Ch54Ord22Level2FreqUseGpuFalse(testCase)
            
            % Parameters
            height = 108;
            width = 135;
            srcImg = rand(height,width);
            nDecs = [ 3 3 ];
            useGpu = false;
            analysisFilters(:,:,1) = randn(9,9);
            analysisFilters(:,:,2) = randn(9,9);
            analysisFilters(:,:,3) = randn(9,9);
            analysisFilters(:,:,4) = randn(9,9);
            analysisFilters(:,:,5) = randn(9,9);
            analysisFilters(:,:,6) = randn(9,9);
            analysisFilters(:,:,7) = randn(9,9);
            analysisFilters(:,:,8) = randn(9,9);
            analysisFilters(:,:,9) = randn(9,9);
            nLevels = 2;
            
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
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
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'UseGpu',useGpu,...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            
        end        
                 
        % Test
        function testClone(testCase)
            
            % Parameters
            height = 108;
            width = 135;
            srcImg = rand(height,width);
            nDecs = [ 3 3 ];
            analysisFilters(:,:,1) = randn(9,9);
            analysisFilters(:,:,2) = randn(9,9);
            analysisFilters(:,:,3) = randn(9,9);
            analysisFilters(:,:,4) = randn(9,9);
            analysisFilters(:,:,5) = randn(9,9);
            analysisFilters(:,:,6) = randn(9,9);
            analysisFilters(:,:,7) = randn(9,9);
            analysisFilters(:,:,8) = randn(9,9);
            analysisFilters(:,:,9) = randn(9,9);
            nLevels = 2;
            
            import saivdr.dictionary.utility.Direction
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},analysisFilters(:,:,iSubband),...
                    'conv','circ').',nDecs(Direction.VERTICAL)).',...
                    nDecs(Direction.HORIZONTAL));
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
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            cloneAnalyzer  = clone(testCase.analyzer);
            
            % Actual values
            [coefsActual,scalesActual] = step(cloneAnalyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            
        end            
    end
    
end
