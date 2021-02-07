classdef Analysis2plus1dSystemTestCase < matlab.unittest.TestCase
    %ANALYSIS2PLUS1DSYSTEMTESTCASE Test case for Analysis2plus1dSystem
    %
    % Requirements: MATLAB R2020b
    %
    % Copyright (c) 2021, Shogo MURAMATSU
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
        %datatype = { 'single', 'double' };
        nsubrows = struct('small', 4,'medium', 8, 'large', 16);
        nsubcols = struct('small', 4,'medium', 8, 'large', 16);
        nsublays = { 1, 2 }; 
        ndecsX = { 1, 2 };
        ndecsY = { 1, 2 };
        ndecsZ = { 2, 4 };
        pordXY = { 0, 4 };
        pordZ = { 0, 2 };
        redundancy = { 1, 2 };        
        nlevels = { 1, 3, 5 };
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
            import saivdr.dictionary.generalfb.*
            analysisFiltersInXYExpctd = 1;
            analysisFiltersInZExpctd = 1;
            decimationFactorExpctd =  [ 2 2 2 ];
            filterDomainExpctd = 'Spatial';
            boundaryOperationExpctd = 'Circular';
            
            % Instantiation
            testCase.analyzer = Analysis2plus1dSystem();
            
            % Actual value
            analysisFiltersInXYActual = get(testCase.analyzer,'AnalysisFiltersInXY');
            analysisFiltersInZActual = get(testCase.analyzer,'AnalysisFiltersInZ');
            decimationFactorActual = get(testCase.analyzer,'DecimationFactor');
            filterDomainActual = get(testCase.analyzer,'FilterDomain');
            boundaryOperationActual = get(testCase.analyzer,'BoundaryOperation');            
            
            % Evaluation
            testCase.assertEqual(analysisFiltersInXYActual,analysisFiltersInXYExpctd);
            testCase.assertEqual(analysisFiltersInZActual,analysisFiltersInZExpctd);
            testCase.assertEqual(decimationFactorActual,decimationFactorExpctd);
            testCase.assertEqual(filterDomainActual,filterDomainExpctd);
            testCase.assertEqual(boundaryOperationActual,boundaryOperationExpctd);  
            
        end
        
        % Test
        function testAnalysisFilters(testCase)
            
            % Expected values
            analysisFiltersInXYExpctd(:,:,1) = randn(2,2);
            analysisFiltersInXYExpctd(:,:,2) = randn(2,2);
            analysisFiltersInXYExpctd(:,:,3) = randn(2,2);
            analysisFiltersInXYExpctd(:,:,4) = randn(2,2);
            analysisFiltersInZExpctd(:,1) = randn(2,1);
            analysisFiltersInZExpctd(:,2) = randn(2,1);
                        
            % Instantiation
            import saivdr.dictionary.generalfb.*
            testCase.analyzer = Analysis2plus1dSystem(...
                'AnalysisFiltersInXY',analysisFiltersInXYExpctd,...
                'AnalysisFiltersInZ', analysisFiltersInZExpctd);
            
            % Actual value
            analysisFiltersXYActual = get(testCase.analyzer,'AnalysisFiltersInXY');
            analysisFiltersZActual = get(testCase.analyzer,'AnalysisFiltersInZ');
            
            % Evaluation
            nChsXY = size(analysisFiltersInXYExpctd,3);
            for iCh = 1:nChsXY
                testCase.assertEqual(analysisFiltersXYActual(:,:,iCh),...
                    analysisFiltersInXYExpctd(:,:,iCh));
            end
            nChsZ = size(analysisFiltersInZExpctd,2);
            for iCh = 1:nChsZ
                testCase.assertEqual(analysisFiltersZActual(:,iCh),...
                    analysisFiltersInZExpctd(:,iCh));
            end
        end

        % Test
        function testStepDecXYLevel1(testCase,...
                nsubrows,nsubcols,nsublays,ndecsX,ndecsY,ndecsZ,pordXY,pordZ,redundancy) 

            % Parameters
            import saivdr.dictionary.utility.Direction
            nDecs = [ ndecsY ndecsX ndecsZ ];                        
            height = nsubrows * ndecsY;
            width = nsubcols * ndecsX;
            depth = nsublays * ndecsZ;
            srcImg = rand(height,width,depth);

            % Filters in XY
            nChsXY = redundancy*ndecsY*ndecsX;
            lenY = (pordXY+1)*ndecsY;
            lenX = (pordXY+1)*ndecsX;
            analysisFiltersInXY = zeros(lenY,lenX,nChsXY);
            for iChXY = 1:nChsXY
                analysisFiltersInXY(:,:,iChXY) = randn(lenY,lenX);
            end
            % Filters in Z
            nChsZ = ndecsZ;
            lenZ = (pordZ+1)*ndecsZ;                                        
            analysisFiltersInZ = zeros(lenZ,nChsZ);
            for iChInZ = 1:nChsZ
                analysisFiltersInZ(:,iChInZ) = randn(lenZ,1);
            end          
            % Tree level
            nLevelsXY = 1;
            
            % Expected values
            import saivdr.dictionary.generalfb.*
            nChs = nChsXY * nChsZ;
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            %
            iCh = 1;
            for iSubbandZ = 1:nChsZ
                % Decimation in Z
                hz = analysisFiltersInZ(:,iSubbandZ);                
                tmpImg = permute(srcImg,[3,1,2]);
                if size(tmpImg,1) == 1
                    subImgZ = imfilter(tmpImg,hz,'conv','circ');
                else
                    subImgZ = downsample(imfilter(tmpImg,hz,'conv','circ'),...
                        nDecs(Direction.DEPTH));
                end
                subImgZ = ipermute(subImgZ,[3,1,2]);
                % Decimation in X-Y
                for iSubbandXY = 1:nChsXY
                    hxy = analysisFiltersInXY(:,:,iSubbandXY);                    
                    % Filter in XY
                    subImgXYZ = imfilter(subImgZ,hxy,'conv','circ');
                    % Downsample in Y
                    if size(subImgXYZ,1) > 1
                        u = downsample(subImgXYZ,...
                            nDecs(Direction.VERTICAL));
                    else
                        u = subImgXYZ;
                    end
                    % Downsample in X                    
                    if size(u,2) > 1
                        subCoefs = ipermute(downsample(permute(u,...
                            [2,1,3]),nDecs(Direction.HORIZONTAL)),[2,1,3]);
                    end
                    coefsExpctd((iCh-1)*nSubCoefs+1:iCh*nSubCoefs) = ...
                        subCoefs(:).';
                    %
                    iCh = iCh + 1;
                end
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = Analysis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFiltersInXY',analysisFiltersInXY,...
                'AnalysisFiltersInZ',analysisFiltersInZ,...
                'NumberOfLevelsInXY',nLevelsXY);
            
            % Actual values
            [coefsActual, scalesActual] = testCase.analyzer.step(srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
                sprintf('%g',diff));

        end
        
        % Test
        function testStepLevel2(testCase,...
                nsubrows,nsubcols,nsublays,ndecsX,ndecsY,ndecsZ,pordXY,pordZ,redundancy)            

            % Parameters
            import saivdr.dictionary.utility.Direction
            nLevelsXY = 2;
            nDecs = [ ndecsY ndecsX ndecsZ ];                        
            height = nsubrows * (ndecsY^nLevelsXY);
            width = nsubcols * (ndecsX^nLevelsXY);
            depth = nsublays * ndecsZ;
            srcImg = rand(height,width,depth);

            % Filters in XY
            nChsXY = redundancy*ndecsY*ndecsX;
            lenY = (pordXY+1)*ndecsY;
            lenX = (pordXY+1)*ndecsX;
            analysisFiltersInXY = zeros(lenY,lenX,nChsXY);
            for iChXY = 1:nChsXY
                analysisFiltersInXY(:,:,iChXY) = randn(lenY,lenX);
            end
            % Filters in Z
            nChsZ = ndecsZ;
            lenZ = (pordZ+1)*ndecsZ;                                        
            analysisFiltersInZ = zeros(lenZ,nChsZ);
            for iChInZ = 1:nChsZ
                analysisFiltersInZ(:,iChInZ) = randn(lenZ,1);
            end          
            
            % Expected values
            import saivdr.dictionary.generalfb.*
            downsample2_ = @(x,d) ... % XY downsampling for 3-D data
                ipermute(downsample(...
                permute(downsample(x,d(1)),[2,1,3]),d(2)),[2,1,3]);
            %
            subCoefs = cell(nChsXY,nLevelsXY,nChsZ);
            for iSubbandZ = 1:nChsZ
                % Decimation in Z
                hz = analysisFiltersInZ(:,iSubbandZ);                
                tmpImg = permute(srcImg,[3,1,2]);                
                if ismatrix(srcImg)
                    subImgZ = imfilter(tmpImg,hz,'conv','circ');
                else
                    subImgZ = downsample(...
                        imfilter(tmpImg,hz,'conv','circ'),...
                        nDecs(Direction.DEPTH));
                end
                subImgZ = ipermute(subImgZ,[3,1,2]);                    
                % Decimation in X-Y
                iLvXY = 1;
                for iSubbandXY = 1:nChsXY
                    hxy = analysisFiltersInXY(:,:,iSubbandXY);                    
                    % Filter in XY
                    subImgXYZ = imfilter(subImgZ,hxy,'conv','circ');
                    % Downsample in XY
                    subCoefs{iSubbandXY,iLvXY,iSubbandZ} = downsample2_(...
                        subImgXYZ,nDecs(Direction.VERTICAL:Direction.HORIZONTAL));
                    %
                end
                iLvXY = iLvXY + 1;
                for iSubbandXY = 1:nChsXY
                    hxy = analysisFiltersInXY(:,:,iSubbandXY);                    
                    % Filter in XY
                    subImgXYZ = imfilter(subCoefs{1,iLvXY-1,iSubbandZ},hxy,'conv','circ');
                    % Downsample in XY
                    subCoefs{iSubbandXY,iLvXY,iSubbandZ} = downsample2_(...
                        subImgXYZ,nDecs(Direction.VERTICAL:Direction.HORIZONTAL));
                    %                    
                end
            end
            % Coefs.& scales
            iCh = 1;
            for iSubbandZ = 1:nChsZ
                for iSubbandXY = 1:nChsXY
                    coefs{iCh} = subCoefs{iSubbandXY,nLevelsXY,iSubbandZ};
                    iCh = iCh + 1;
                end                
                for iLvXY = nLevelsXY-1:-1:1
                    for iSubbandXY = 2:nChsXY
                        coefs{iCh} = subCoefs{iSubbandXY,iLvXY,iSubbandZ};
                        iCh = iCh + 1;
                    end
                end
            end
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                tmpCoefs = coefs{iSubband};
                if ismatrix(tmpCoefs)
                    scalesExpctd(iSubband,:) = [ size(tmpCoefs) 1 ];
                else
                    scalesExpctd(iSubband,:) = size(tmpCoefs);
                end
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end

             % Instantiation of target class
            testCase.analyzer = Analysis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFiltersInXY',analysisFiltersInXY,...
                'AnalysisFiltersInZ',analysisFiltersInZ,...
                'NumberOfLevelsInXY',nLevelsXY);
            
            % Actual values
            [coefsActual, scalesActual] = testCase.analyzer.step(srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
                sprintf('%g',diff));

        end

         % Test
        function testStepLevelN(testCase,ndecsX,ndecsY,ndecsZ,nlevels)            

            redundancy_ = 2;
            nsubrows_ = 4;
            nsubcols_ = 2;
            nsublays_ = 1;
            pordXY_ = 4;
            pordZ_ = 0;
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nLevelsXY = nlevels;
            nDecs = [ ndecsY ndecsX ndecsZ ];                        
            height = nsubrows_ * (ndecsY^nLevelsXY);
            width = nsubcols_ * (ndecsX^nLevelsXY);
            depth = nsublays_ * ndecsZ;
            srcImg = rand(height,width,depth);

            % Filters in XY
            nChsXY = redundancy_*ndecsY*ndecsX;
            lenY = (pordXY_+1)*ndecsY;
            lenX = (pordXY_+1)*ndecsX;
            analysisFiltersInXY = zeros(lenY,lenX,nChsXY);
            for iChXY = 1:nChsXY
                analysisFiltersInXY(:,:,iChXY) = randn(lenY,lenX);
            end
            % Filters in Z
            nChsZ = ndecsZ;
            lenZ = (pordZ_+1)*ndecsZ;                                        
            analysisFiltersInZ = zeros(lenZ,nChsZ);
            for iChInZ = 1:nChsZ
                analysisFiltersInZ(:,iChInZ) = randn(lenZ,1);
            end          
            
            % Expected values
            import saivdr.dictionary.generalfb.*
            downsample2_ = @(x,d) ... % XY downsampling for 3-D data
                ipermute(downsample(...
                permute(downsample(x,d(1)),[2,1,3]),d(2)),[2,1,3]);
            %
            subCoefs = cell(nChsXY,nLevelsXY,nChsZ);
            for iSubbandZ = 1:nChsZ
                % Decimation in Z
                hz = analysisFiltersInZ(:,iSubbandZ);                
                tmpImg = permute(srcImg,[3,1,2]);                
                if ismatrix(srcImg)
                    subImgZ = imfilter(tmpImg,hz,'conv','circ');
                else
                    subImgZ = downsample(...
                        imfilter(tmpImg,hz,'conv','circ'),...
                        nDecs(Direction.DEPTH));
                end
                subImgZ = ipermute(subImgZ,[3,1,2]);                    
                % Decimation in X-Y
                for iLvXY = 1:nLevelsXY
                    for iSubbandXY = 1:nChsXY
                        hxy = analysisFiltersInXY(:,:,iSubbandXY);
                        % Filter in XY
                        subImgXYZ = imfilter(subImgZ,hxy,'conv','circ');
                        % Downsample in XY
                        subCoefs{iSubbandXY,iLvXY,iSubbandZ} = downsample2_(...
                            subImgXYZ,nDecs(Direction.VERTICAL:Direction.HORIZONTAL));
                        %
                         
                    end
                    subImgZ = subCoefs{1,iLvXY,iSubbandZ};
                end
            end
            % Coefs.& scales
            iCh = 1;
            for iSubbandZ = 1:nChsZ
                for iSubbandXY = 1:nChsXY
                    coefs{iCh} = subCoefs{iSubbandXY,nLevelsXY,iSubbandZ};
                    iCh = iCh + 1;
                end                
                for iLvXY = nLevelsXY-1:-1:1
                    for iSubbandXY = 2:nChsXY
                        coefs{iCh} = subCoefs{iSubbandXY,iLvXY,iSubbandZ};
                        iCh = iCh + 1;
                    end
                end
            end
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                tmpCoefs = coefs{iSubband};
                if ismatrix(tmpCoefs)
                    scalesExpctd(iSubband,:) = [ size(tmpCoefs) 1 ];
                else
                    scalesExpctd(iSubband,:) = size(tmpCoefs);
                end
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end

             % Instantiation of target class
            testCase.analyzer = Analysis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFiltersInXY',analysisFiltersInXY,...
                'AnalysisFiltersInZ',analysisFiltersInZ,...
                'NumberOfLevelsInXY',nLevelsXY);
            
            % Actual values
            [coefsActual, scalesActual] = testCase.analyzer.step(srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
                sprintf('%g',diff));

        end

        %{
        % Test
        function testStepDec221Ch54Ord000Level1(testCase)

            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 1 ]; 
            analysisFilters(:,:,1) = randn(2,2);
            analysisFilters(:,:,2) = randn(2,2);
            analysisFilters(:,:,3) = randn(2,2);
            analysisFilters(:,:,4) = randn(2,2);
            analysisFilters(:,:,5) = randn(2,2);
            analysisFilters(:,:,6) = randn(2,2);
            analysisFilters(:,:,7) = randn(2,2);
            analysisFilters(:,:,8) = randn(2,2);
            analysisFilters(:,:,9) = randn(2,2);
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);  
            % DCT for Z direction
            dctImg = zeros(height,width,depth);
            for iCol = 1:width
                for iRow = 1:height
                    dctImg(iRow,iCol,:) = dct(srcImg(iRow,iCol,:));
                end
            end
            % X-Y filtering
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);                
                filtImg = zeros(height,width,depth);
                for iLay = 1:depth
                    filtImg(:,:,iLay) = imfilter(dctImg(:,:,iLay),h,...
                        'conv','circ');
                end
                subCoef = downsample3_(filtImg,nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = Analysis2plus1dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
                sprintf('%g',diff));

        end

        % Test        
        function testStepDec221Ch54Ord222Level1(testCase)

            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 1 ]; 
            analysisFilters(:,:,1) = randn(6,6);
            analysisFilters(:,:,2) = randn(6,6);
            analysisFilters(:,:,3) = randn(6,6);
            analysisFilters(:,:,4) = randn(6,6);
            analysisFilters(:,:,5) = randn(6,6);
            analysisFilters(:,:,6) = randn(6,6);
            analysisFilters(:,:,7) = randn(6,6);
            analysisFilters(:,:,8) = randn(6,6);
            analysisFilters(:,:,9) = randn(6,6);
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            % DCT for Z direction
            dctImg = zeros(height,width,depth);
            for iCol = 1:width
                for iRow = 1:height
                    dctImg(iRow,iCol,:) = dct(srcImg(iRow,iCol,:));
                end
            end
            % X-Y filtering
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);                
                filtImg = zeros(height,width,depth);
                for iLay = 1:depth
                    filtImg(:,:,iLay) = imfilter(dctImg(:,:,iLay),h,...
                        'conv','circ');
                end
                subCoef = downsample3_(filtImg,nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = Analysis2plus1dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,1) = randn(2,2);
            analysisFilters(:,:,2) = randn(2,2);
            analysisFilters(:,:,3) = randn(2,2);
            analysisFilters(:,:,4) = randn(2,2);
            analysisFilters(:,:,5) = randn(2,2);
            analysisFilters(:,:,6) = randn(2,2);
            analysisFilters(:,:,7) = randn(2,2);
            analysisFilters(:,:,8) = randn(2,2);
            analysisFilters(:,:,9) = randn(2,2);
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            % DCT for Z direction
            dctImg = zeros(height,width,depth);
            for iCol = 1:width
                for iRow = 1:height
                    dctImg(iRow,iCol,:) = dct(srcImg(iRow,iCol,:));
                end
            end
            % X-Y filtering
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);                
                filtImg = zeros(height,width,depth);
                for iLay = 1:depth
                    filtImg(:,:,iLay) = imfilter(dctImg(:,:,iLay),h,...
                        'conv','circ');
                end
                subCoef = downsample3_(filtImg,nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = Analysis2plus1dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,1) = randn(9,6);
            analysisFilters(:,:,2) = randn(9,6);
            analysisFilters(:,:,3) = randn(9,6);
            analysisFilters(:,:,4) = randn(9,6);
            analysisFilters(:,:,5) = randn(9,6);
            analysisFilters(:,:,6) = randn(9,6);
            analysisFilters(:,:,7) = randn(9,6);
            analysisFilters(:,:,8) = randn(9,6);
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            % DCT for Z direction
            dctImg = zeros(height,width,depth);
            for iCol = 1:width
                for iRow = 1:height
                    dctImg(iRow,iCol,:) = dct(srcImg(iRow,iCol,:));
                end
            end
            % X-Y filtering
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);                
                filtImg = zeros(height,width,depth);
                for iLay = 1:depth
                    filtImg(:,:,iLay) = imfilter(dctImg(:,:,iLay),h,...
                        'conv','circ');
                end
                subCoef = downsample3_(filtImg,nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = Analysis2plus1dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
                sprintf('%g',diff));

        end
        
        % Test
        function testStepDec221Ch44Ord222Level2(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 1 ]; 
            analysisFilters(:,:,1) = randn(6,6);
            analysisFilters(:,:,2) = randn(6,6);
            analysisFilters(:,:,3) = randn(6,6);
            analysisFilters(:,:,4) = randn(6,6);
            analysisFilters(:,:,5) = randn(6,6);
            analysisFilters(:,:,6) = randn(6,6);
            analysisFilters(:,:,7) = randn(6,6);
            analysisFilters(:,:,8) = randn(6,6);        
            nLevels = 2;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);              
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            % DCT for Z direction (Lv1)
            dctImgLv1 = zeros(height,width,depth);
            for iCol = 1:width
                for iRow = 1:height
                    dctImgLv1(iRow,iCol,:) = dct(srcImg(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv1)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv1 = zeros(height,width,depth);
                for iLay = 1:depth
                    filtImgLv1(:,:,iLay) = imfilter(dctImgLv1(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv1{iSubband} = downsample3_(filtImgLv1,nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            % DCT for Z direction (Lv2)
            dctImgLv2 = zeros([height,width,depth]./nDecs);
            for iCol = 1:width/nDecs(2)
                for iRow = 1:height/nDecs(1)
                    dctImgLv2(iRow,iCol,:) = dct(coefsExpctdLv1{1}(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv2)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv2 = zeros([height,width,depth]./nDecs);
                for iLay = 1:depth/nDecs(3)
                    filtImgLv2(:,:,iLay) = imfilter(dctImgLv2(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv2{iSubband} = downsample3_(filtImgLv2,nDecs);
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
            testCase.analyzer = Analysis2plus1dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
                sprintf('%g',diff));  
            
        end
        
        % Test
        function testStepDec221Ch44Ord222Level3(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 1 ];
            analysisFilters(:,:,1) = randn(6,6);
            analysisFilters(:,:,2) = randn(6,6);
            analysisFilters(:,:,3) = randn(6,6);
            analysisFilters(:,:,4) = randn(6,6);
            analysisFilters(:,:,5) = randn(6,6);
            analysisFilters(:,:,6) = randn(6,6);
            analysisFilters(:,:,7) = randn(6,6);
            analysisFilters(:,:,8) = randn(6,6);
            nLevels = 3;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            % DCT for Z direction (Lv1)
            dctImgLv1 = zeros(height,width,depth);
            for iCol = 1:width
                for iRow = 1:height
                    dctImgLv1(iRow,iCol,:) = dct(srcImg(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv1)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv1 = zeros(height,width,depth);
                for iLay = 1:depth
                    filtImgLv1(:,:,iLay) = imfilter(dctImgLv1(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv1{iSubband} = downsample3_(filtImgLv1,nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            % DCT for Z direction (Lv2)
            dctImgLv2 = zeros([height,width,depth]./nDecs);
            for iCol = 1:width/nDecs(2)
                for iRow = 1:height/nDecs(1)
                    dctImgLv2(iRow,iCol,:) = dct(coefsExpctdLv1{1}(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv2)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv2 = zeros([height,width,depth]./nDecs);
                for iLay = 1:depth/nDecs(3)
                    filtImgLv2(:,:,iLay) = imfilter(dctImgLv2(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv2{iSubband} = downsample3_(filtImgLv2,nDecs);
            end            
            coefsExpctdLv3 = cell(nChs,1);
            % DCT for Z direction (Lv3)
            dctImgLv3 = zeros([height,width,depth]./nDecs.^2);
            for iCol = 1:width/nDecs(2)^2
                for iRow = 1:height/nDecs(1)^2
                    dctImgLv3(iRow,iCol,:) = dct(coefsExpctdLv2{1}(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv3)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv3 = zeros([height,width,depth]./nDecs.^2);
                for iLay = 1:depth/nDecs(3)^2
                    filtImgLv3(:,:,iLay) = imfilter(dctImgLv3(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv3{iSubband} = downsample3_(filtImgLv3,nDecs);
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
            testCase.analyzer = Analysis2plus1dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec221Ch55Ord444Level3(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 2 1 ];
            analysisFilters(:,:,1) = randn(10,10);
            analysisFilters(:,:,2) = randn(10,10);
            analysisFilters(:,:,3) = randn(10,10);
            analysisFilters(:,:,4) = randn(10,10);
            analysisFilters(:,:,5) = randn(10,10);
            analysisFilters(:,:,6) = randn(10,10);
            analysisFilters(:,:,7) = randn(10,10);
            analysisFilters(:,:,8) = randn(10,10);
            analysisFilters(:,:,9) = randn(10,10);
            analysisFilters(:,:,10) = randn(10,10);
            nLevels = 3;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            % DCT for Z direction (Lv1)
            dctImgLv1 = zeros(height,width,depth);
            for iCol = 1:width
                for iRow = 1:height
                    dctImgLv1(iRow,iCol,:) = dct(srcImg(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv1)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv1 = zeros(height,width,depth);
                for iLay = 1:depth
                    filtImgLv1(:,:,iLay) = imfilter(dctImgLv1(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv1{iSubband} = downsample3_(filtImgLv1,nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            % DCT for Z direction (Lv2)
            dctImgLv2 = zeros([height,width,depth]./nDecs);
            for iCol = 1:width/nDecs(2)
                for iRow = 1:height/nDecs(1)
                    dctImgLv2(iRow,iCol,:) = dct(coefsExpctdLv1{1}(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv2)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv2 = zeros([height,width,depth]./nDecs);
                for iLay = 1:depth/nDecs(3)
                    filtImgLv2(:,:,iLay) = imfilter(dctImgLv2(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv2{iSubband} = downsample3_(filtImgLv2,nDecs);
            end            
            coefsExpctdLv3 = cell(nChs,1);
            % DCT for Z direction (Lv3)
            dctImgLv3 = zeros([height,width,depth]./nDecs.^2);
            for iCol = 1:width/nDecs(2)^2
                for iRow = 1:height/nDecs(1)^2
                    dctImgLv3(iRow,iCol,:) = dct(coefsExpctdLv2{1}(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv3)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv3 = zeros([height,width,depth]./nDecs.^2);
                for iLay = 1:depth/nDecs(3)^2
                    filtImgLv3(:,:,iLay) = imfilter(dctImgLv3(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv3{iSubband} = downsample3_(filtImgLv3,nDecs);
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
            testCase.analyzer = Analysis2plus1dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,1) = randn(2,2);
            analysisFilters(:,:,2) = randn(2,2);
            analysisFilters(:,:,3) = randn(2,2);
            analysisFilters(:,:,4) = randn(2,2);
            analysisFilters(:,:,5) = randn(2,2);
            analysisFilters(:,:,6) = randn(2,2);
            analysisFilters(:,:,7) = randn(2,2);
            analysisFilters(:,:,8) = randn(2,2);
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.generalfb.*
            nChs = size(analysisFilters,3);
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);            
            % DCT for Z direction
            dctImg = zeros(height,width,depth);
            for iCol = 1:width
                for iRow = 1:height
                    dctImg(iRow,iCol,:) = dct(srcImg(iRow,iCol,:));
                end
            end
            % X-Y filtering
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImg = zeros(height,width,depth);
                for iLay = 1:depth
                    filtImg(:,:,iLay) = imfilter(dctImg(:,:,iLay),h,...
                        'conv','circ');
                end
                subCoef = downsample3_(filtImg,nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,nChs,1);

            % Instantiation of target class
            testCase.analyzer = Analysis2plus1dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,:,1) = randn(2,2,2);
            analysisFilters(:,:,:,2) = randn(2,2,2);
            analysisFilters(:,:,:,3) = randn(2,2,2);
            analysisFilters(:,:,:,4) = randn(2,2,2);
            analysisFilters(:,:,:,5) = randn(2,2,2);
            analysisFilters(:,:,:,6) = randn(2,2,2);
            analysisFilters(:,:,:,7) = randn(2,2,2);
            analysisFilters(:,:,:,8) = randn(2,2,2);
            analysisFilters(:,:,:,9) = randn(2,2,2);
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
            testCase.analyzer = Analysis3dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,:,1) = randn(6,6,6);
            analysisFilters(:,:,:,2) = randn(6,6,6);
            analysisFilters(:,:,:,3) = randn(6,6,6);
            analysisFilters(:,:,:,4) = randn(6,6,6);
            analysisFilters(:,:,:,5) = randn(6,6,6);
            analysisFilters(:,:,:,6) = randn(6,6,6);
            analysisFilters(:,:,:,7) = randn(6,6,6);
            analysisFilters(:,:,:,8) = randn(6,6,6);
            analysisFilters(:,:,:,9) = randn(6,6,6);
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
            testCase.analyzer = Analysis3dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,:,1) = randn(2,2,2);
            analysisFilters(:,:,:,2) = randn(2,2,2);
            analysisFilters(:,:,:,3) = randn(2,2,2);
            analysisFilters(:,:,:,4) = randn(2,2,2);
            analysisFilters(:,:,:,5) = randn(2,2,2);
            analysisFilters(:,:,:,6) = randn(2,2,2);
            analysisFilters(:,:,:,7) = randn(2,2,2);
            analysisFilters(:,:,:,8) = randn(2,2,2);
            analysisFilters(:,:,:,9) = randn(2,2,2);
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
            testCase.analyzer = Analysis3dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,:,1) = randn(9,6,3);
            analysisFilters(:,:,:,2) = randn(9,6,3);
            analysisFilters(:,:,:,3) = randn(9,6,3);
            analysisFilters(:,:,:,4) = randn(9,6,3);
            analysisFilters(:,:,:,5) = randn(9,6,3);
            analysisFilters(:,:,:,6) = randn(9,6,3);
            analysisFilters(:,:,:,7) = randn(9,6,3);
            analysisFilters(:,:,:,8) = randn(9,6,3);
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
            testCase.analyzer = Analysis3dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,:,1) = randn(6,6,6);
            analysisFilters(:,:,:,2) = randn(6,6,6);
            analysisFilters(:,:,:,3) = randn(6,6,6);
            analysisFilters(:,:,:,4) = randn(6,6,6);
            analysisFilters(:,:,:,5) = randn(6,6,6);
            analysisFilters(:,:,:,6) = randn(6,6,6);
            analysisFilters(:,:,:,7) = randn(6,6,6);
            analysisFilters(:,:,:,8) = randn(6,6,6);        
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
            testCase.analyzer = Analysis3dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,:,1) = randn(6,6,6);
            analysisFilters(:,:,:,2) = randn(6,6,6);
            analysisFilters(:,:,:,3) = randn(6,6,6);
            analysisFilters(:,:,:,4) = randn(6,6,6);
            analysisFilters(:,:,:,5) = randn(6,6,6);
            analysisFilters(:,:,:,6) = randn(6,6,6);
            analysisFilters(:,:,:,7) = randn(6,6,6);
            analysisFilters(:,:,:,8) = randn(6,6,6);
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
            testCase.analyzer = Analysis3dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,:,1) = randn(10,10,10);
            analysisFilters(:,:,:,2) = randn(10,10,10);
            analysisFilters(:,:,:,3) = randn(10,10,10);
            analysisFilters(:,:,:,4) = randn(10,10,10);
            analysisFilters(:,:,:,5) = randn(10,10,10);
            analysisFilters(:,:,:,6) = randn(10,10,10);
            analysisFilters(:,:,:,7) = randn(10,10,10);
            analysisFilters(:,:,:,8) = randn(10,10,10);
            analysisFilters(:,:,:,9) = randn(10,10,10);
            analysisFilters(:,:,:,10) = randn(10,10,10);
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
            testCase.analyzer = Analysis3dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,1) = randn(9,6);
            analysisFilters(:,:,2) = randn(9,6);
            analysisFilters(:,:,3) = randn(9,6);
            analysisFilters(:,:,4) = randn(9,6);
            analysisFilters(:,:,5) = randn(9,6);
            analysisFilters(:,:,6) = randn(9,6);
            analysisFilters(:,:,7) = randn(9,6);
            analysisFilters(:,:,8) = randn(9,6);
            nLevels = 2;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            % DCT for Z direction (Lv1)
            dctImgLv1 = zeros(height,width,depth);
            for iCol = 1:width
                for iRow = 1:height
                    dctImgLv1(iRow,iCol,:) = dct(srcImg(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv1)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv1 = zeros(height,width,depth);
                for iLay = 1:depth
                    filtImgLv1(:,:,iLay) = imfilter(dctImgLv1(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv1{iSubband} = downsample3_(filtImgLv1,nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            % DCT for Z direction (Lv2)
            dctImgLv2 = zeros([height,width,depth]./nDecs);
            for iCol = 1:width/nDecs(2)
                for iRow = 1:height/nDecs(1)
                    dctImgLv2(iRow,iCol,:) = dct(coefsExpctdLv1{1}(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv2)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv2 = zeros([height,width,depth]./nDecs);
                for iLay = 1:depth/nDecs(3)
                    filtImgLv2(:,:,iLay) = imfilter(dctImgLv2(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv2{iSubband} = downsample3_(filtImgLv2,nDecs);
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
            testCase.analyzer = Analysis2plus1dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,:,1) = randn(9,6,3);
            analysisFilters(:,:,:,2) = randn(9,6,3);
            analysisFilters(:,:,:,3) = randn(9,6,3);
            analysisFilters(:,:,:,4) = randn(9,6,3);
            analysisFilters(:,:,:,5) = randn(9,6,3);
            analysisFilters(:,:,:,6) = randn(9,6,3);
            analysisFilters(:,:,:,7) = randn(9,6,3);
            analysisFilters(:,:,:,8) = randn(9,6,3);
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
            testCase.analyzer = Analysis3dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,1) = randn(9,6);
            analysisFilters(:,:,2) = randn(9,6);
            analysisFilters(:,:,3) = randn(9,6);
            analysisFilters(:,:,4) = randn(9,6);
            analysisFilters(:,:,5) = randn(9,6);
            analysisFilters(:,:,6) = randn(9,6);
            analysisFilters(:,:,7) = randn(9,6);
            analysisFilters(:,:,8) = randn(9,6);
            nLevels = 3;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            % DCT for Z direction (Lv1)
            dctImgLv1 = zeros(height,width,depth);
            for iCol = 1:width
                for iRow = 1:height
                    dctImgLv1(iRow,iCol,:) = dct(srcImg(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv1)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv1 = zeros(height,width,depth);
                for iLay = 1:depth
                    filtImgLv1(:,:,iLay) = imfilter(dctImgLv1(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv1{iSubband} = downsample3_(filtImgLv1,nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            % DCT for Z direction (Lv2)
            dctImgLv2 = zeros([height,width,depth]./nDecs);
            for iCol = 1:width/nDecs(2)
                for iRow = 1:height/nDecs(1)
                    dctImgLv2(iRow,iCol,:) = dct(coefsExpctdLv1{1}(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv2)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv2 = zeros([height,width,depth]./nDecs);
                for iLay = 1:depth/nDecs(3)
                    filtImgLv2(:,:,iLay) = imfilter(dctImgLv2(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv2{iSubband} = downsample3_(filtImgLv2,nDecs);
            end            
            coefsExpctdLv3 = cell(nChs,1);
            % DCT for Z direction (Lv3)
            dctImgLv3 = zeros([height,width,depth]./nDecs.^2);
            for iCol = 1:width/nDecs(2)^2
                for iRow = 1:height/nDecs(1)^2
                    dctImgLv3(iRow,iCol,:) = dct(coefsExpctdLv2{1}(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv3)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv3 = zeros([height,width,depth]./nDecs.^2);
                for iLay = 1:depth/nDecs(3)^2
                    filtImgLv3(:,:,iLay) = imfilter(dctImgLv3(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv3{iSubband} = downsample3_(filtImgLv3,nDecs);
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
            testCase.analyzer = Analysis2plus1dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,:,1) = randn(9,6,3);
            analysisFilters(:,:,:,2) = randn(9,6,3);
            analysisFilters(:,:,:,3) = randn(9,6,3);
            analysisFilters(:,:,:,4) = randn(9,6,3);
            analysisFilters(:,:,:,5) = randn(9,6,3);
            analysisFilters(:,:,:,6) = randn(9,6,3);
            analysisFilters(:,:,:,7) = randn(9,6,3);
            analysisFilters(:,:,:,8) = randn(9,6,3);
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
            testCase.analyzer = Analysis3dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec231Ch1414Ord222Level2(testCase) 
            
            % Parameters
            height = 8*2^2;
            width = 12*3^2;
            depth = 16*4^2;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 3 1 ];
            analysisFilters = zeros(6,9,28);
            for iCh = 1:28
                analysisFilters(:,:,iCh) = randn(6,9);
            end
            nLevels = 2;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            % DCT for Z direction (Lv1)
            dctImgLv1 = zeros(height,width,depth);
            for iCol = 1:width
                for iRow = 1:height
                    dctImgLv1(iRow,iCol,:) = dct(srcImg(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv1)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv1 = zeros(height,width,depth);
                for iLay = 1:depth
                    filtImgLv1(:,:,iLay) = imfilter(dctImgLv1(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv1{iSubband} = downsample3_(filtImgLv1,nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            % DCT for Z direction (Lv2)
            dctImgLv2 = zeros([height,width,depth]./nDecs);
            for iCol = 1:width/nDecs(2)
                for iRow = 1:height/nDecs(1)
                    dctImgLv2(iRow,iCol,:) = dct(coefsExpctdLv1{1}(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv2)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv2 = zeros([height,width,depth]./nDecs);
                for iLay = 1:depth/nDecs(3)
                    filtImgLv2(:,:,iLay) = imfilter(dctImgLv2(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv2{iSubband} = downsample3_(filtImgLv2,nDecs);
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
            testCase.analyzer = Analysis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Spatial',...
                'NumberOfLevels',nLevels);
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
                analysisFilters(:,:,:,iCh) = randn(6,9,12);
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
            testCase.analyzer = Analysis3dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
                sprintf('%g',diff));
            
        end
        
         % Test
        function testStepDec234Ch1414Ord222Level2FreqGpuFalse(testCase) 
            
            % Parameters
            height = 8*2^2;
            width = 12*3^2;
            depth = 16*4^2;
            srcImg = rand(height,width,depth);
            nDecs = [ 2 3 4 ];
            useGpu = false;
            analysisFilters = zeros(6,9,12,28);
            for iCh = 1:28
                analysisFilters(:,:,:,iCh) = randn(6,9,12);
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
            testCase.analyzer = Analysis3dSystem(...
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
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
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
            analysisFilters(:,:,1) = randn(9,6);
            analysisFilters(:,:,2) = randn(9,6);
            analysisFilters(:,:,3) = randn(9,6);
            analysisFilters(:,:,4) = randn(9,6);
            analysisFilters(:,:,5) = randn(9,6);
            analysisFilters(:,:,6) = randn(9,6);
            analysisFilters(:,:,7) = randn(9,6);
            analysisFilters(:,:,8) = randn(9,6);
            nLevels = 3;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            nChs = size(analysisFilters,3);
            coefsExpctdLv1 = cell(nChs,1);
            % DCT for Z direction (Lv1)
            dctImgLv1 = zeros(height,width,depth);
            for iCol = 1:width
                for iRow = 1:height
                    dctImgLv1(iRow,iCol,:) = dct(srcImg(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv1)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv1 = zeros(height,width,depth);
                for iLay = 1:depth
                    filtImgLv1(:,:,iLay) = imfilter(dctImgLv1(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv1{iSubband} = downsample3_(filtImgLv1,nDecs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            % DCT for Z direction (Lv2)
            dctImgLv2 = zeros([height,width,depth]./nDecs);
            for iCol = 1:width/nDecs(2)
                for iRow = 1:height/nDecs(1)
                    dctImgLv2(iRow,iCol,:) = dct(coefsExpctdLv1{1}(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv2)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv2 = zeros([height,width,depth]./nDecs);
                for iLay = 1:depth/nDecs(3)
                    filtImgLv2(:,:,iLay) = imfilter(dctImgLv2(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv2{iSubband} = downsample3_(filtImgLv2,nDecs);
            end            
            coefsExpctdLv3 = cell(nChs,1);
            % DCT for Z direction (Lv3)
            dctImgLv3 = zeros([height,width,depth]./nDecs.^2);
            for iCol = 1:width/nDecs(2)^2
                for iRow = 1:height/nDecs(1)^2
                    dctImgLv3(iRow,iCol,:) = dct(coefsExpctdLv2{1}(iRow,iCol,:));
                end
            end
            % X-Y filtering (Lv3)
            for iSubband = 1:nChs
                h = analysisFilters(:,:,iSubband);
                filtImgLv3 = zeros([height,width,depth]./nDecs.^2);
                for iLay = 1:depth/nDecs(3)^2
                    filtImgLv3(:,:,iLay) = imfilter(dctImgLv3(:,:,iLay),h,...
                        'conv','circ');
                end
                coefsExpctdLv3{iSubband} = downsample3_(filtImgLv3,nDecs);
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
            testCase.analyzer = Analysis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'AnalysisFilters',analysisFilters,...
                'FilterDomain','Frequency',...
                'NumberOfLevels',nLevels);
            cloneAnalyzer = clone(testCase.analyzer);
            
            % Actual values
            [coefsActual,scalesActual] = step(cloneAnalyzer,srcImg);
            
            % Evaluation
            testCase.verifySize(scalesActual,size(scalesExpctd));
            testCase.verifyEqual(scalesActual,scalesExpctd);
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-8,...
                sprintf('%g',diff));
            
        end       
        %}
    end

end
