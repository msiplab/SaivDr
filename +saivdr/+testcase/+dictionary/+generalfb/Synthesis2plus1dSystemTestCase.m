classdef Synthesis2plus1dSystemTestCase < matlab.unittest.TestCase
    %SYNTHESIS2PLUS1DSYSTEMTESTCASE Test case for Synthesis2plus1dSystem
    %
    % Requirements: MATLAB R2015b
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
        synthesizer
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.synthesizer);
        end
    end
    
    methods (Test)
        
        % Test
        function testDefaultConstruction(testCase)
            
            % Expected values
            import saivdr.dictionary.generalfb.*
            synthesisFiltersInXYExpctd = 1;
            synthesisFiltersInZExpctd = 1;
            decimationFactorExpctd = [ 2 2 2 ];
            frmbdExpctd  = [];
            filterDomainExpctd = 'Spatial';
            boundaryOperationExpctd = 'Circular';
            
            % Instantiation
            testCase.synthesizer = Synthesis2plus1dSystem();
            
            % Actual value
            synthesisFiltersInXYActual = get(testCase.synthesizer,'SynthesisFiltersInXY');
            synthesisFiltersInZActual = get(testCase.synthesizer,'SynthesisFiltersInZ');
            decimationFactorActual = get(testCase.synthesizer,'DecimationFactor');
            frmbdActual  = get(testCase.synthesizer,'FrameBound');
            filterDomainActual = get(testCase.synthesizer,'FilterDomain');
            boundaryOperationActual = get(testCase.synthesizer,'BoundaryOperation');  
            
            % Evaluation
            testCase.assertEqual(synthesisFiltersInXYActual,synthesisFiltersInXYExpctd);
            testCase.assertEqual(synthesisFiltersInZActual,synthesisFiltersInZExpctd);
            testCase.assertEqual(decimationFactorActual,decimationFactorExpctd);
            testCase.assertEqual(frmbdActual,frmbdExpctd);
            testCase.assertEqual(filterDomainActual,filterDomainExpctd);
            testCase.assertEqual(boundaryOperationActual,boundaryOperationExpctd);            
        end
       
        % Test
        function testSynthesisFilters(testCase)
            
            % Expected values
            synthesisFiltersInXYExpctd(:,:,1) = randn(2,2);
            synthesisFiltersInXYExpctd(:,:,2) = randn(2,2);
            synthesisFiltersInXYExpctd(:,:,3) = randn(2,2);
            synthesisFiltersInXYExpctd(:,:,4) = randn(2,2);
            synthesisFiltersInZExpctd(:,1) = randn(2,1);
            synthesisFiltersInZExpctd(:,2) = randn(2,1);
            
            % Instantiation
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'SynthesisFiltersInXY',synthesisFiltersInXYExpctd,...
                'SynthesisFiltersInZ',synthesisFiltersInZExpctd);
            
            % Actual value
            synthesisFiltersInXYActual = get(testCase.synthesizer,'SynthesisFiltersInXY');
            synthesisFiltersInZActual = get(testCase.synthesizer,'SynthesisFiltersInZ');
            
            % Evaluation
            nChsXY = size(synthesisFiltersInXYExpctd,3);
            for iCh = 1:nChsXY
                testCase.assertEqual(synthesisFiltersInXYActual(:,:,iCh),...
                    synthesisFiltersInXYExpctd(:,:,iCh));
            end
            nChsZ = size(synthesisFiltersInZExpctd,2);
            for iCh = 1:nChsZ
                testCase.assertEqual(synthesisFiltersInZActual(:,iCh),...
                    synthesisFiltersInZExpctd(:,iCh));
            end
            
        end
        
        % Test
        function testStepLevel1(testCase,...
                nsubrows,nsubcols,nsublays,ndecsX,ndecsY,ndecsZ,pordXY,pordZ)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nDecs = [ ndecsY ndecsX ndecsZ ];
            redundancy_ = 2;

            % Filters in XY
            nChsXY = redundancy_*ndecsY*ndecsX;
            lenY = (pordXY+1)*ndecsY;
            lenX = (pordXY+1)*ndecsX;
            synthesisFiltersInXY = zeros(lenY,lenX,nChsXY);
            for iChXY = 1:nChsXY
                synthesisFiltersInXY(:,:,iChXY) = randn(lenY,lenX);
            end
            % Filters in Z
            nChsZ = ndecsZ;
            lenZ = (pordZ+1)*ndecsZ;                
            synthesisFiltersInZ = zeros(lenZ,nChsZ);
            for iChInZ = 1:nChsZ
                synthesisFiltersInZ(:,iChInZ) = randn(lenZ,1);
            end            
            
            % Expected values
            import saivdr.dictionary.generalfb.*            
            nChs = nChsXY * nChsZ;
            %
            subCoefs = cell(nChs,1);
            subScales = [nsubrows, nsubcols, nsublays];
            for iCh = 1:nChs
                subCoefs{iCh} = randn(subScales);
            end
            coefs = cell2mat(...
                cellfun(@(x) x(:),subCoefs,'UniformOutput',false)).';
            scales = repmat(subScales,nChs,1);  
            %
            phase = 1-mod(nDecs,2); % for phase adjustment required experimentaly
            %
            imgExpctd = 0;
            iCh = 1;
            for iSubbandZ = 1:nChsZ
                subImgInZ = 0;
                % Interpolation in XY
                for iSubbandXY = 1:nChsXY
                    subImgInXY = subCoefs{iCh};
                    fxy = synthesisFiltersInXY(:,:,iSubbandXY);
                    % Upsample in Y
                    if size(subImgInXY,1)==1
                        tmpImg = cat(1,subImgInXY,...
                            zeros(nDecs(1)-1,size(subImgInXY,2),size(subImgInXY,3)));
                        tmpImg = cirshift(tmpImg,[phase(1) 0 0 ]);
                    else
                        tmpImg = upsample(subImgInXY,nDecs(1),phase(1));
                    end
                    if size(tmpImg,2)==1
                        tmpImg = cat(2,tmpImg,...
                            zeros(size(tmpImg,1),nDecs(2)-1,size(tmpImg,3)));
                        tmpImg = cirshift(tmpImg,[0 phase(2) 0 ]);
                    else
                        tmpImg = ipermute(upsample(permute(tmpImg,[2,1,3]),nDecs(2),phase(2)),[2,1,3]);
                    end
                    % Filter in XY
                    subImgInZ = subImgInZ ...
                        + imfilter(tmpImg,fxy,'circ','conv');
                    %
                    iCh = iCh + 1;
                end
                fz = synthesisFiltersInZ(:,iSubbandZ);
                % Interpolation in Z
                if ismatrix(subImgInZ)
                    % Upsample in Z and permute
                    tmpImg = permute(circshift(cat(3,subImgInZ,zeros(size(subImgInZ,1),size(subImgInZ,2),...
                        nDecs(Direction.DEPTH)-1)),[0 0 phase(Direction.DEPTH)]),[3,1,2]);
                else
                    % Upsample in Z and permute                    
                    tmpImg = upsample(permute(subImgInZ,[3,1,2]),...
                        nDecs(Direction.DEPTH),phase(Direction.DEPTH));
                end
                subImgInZ = imfilter(tmpImg,fz,'circ','conv');
                imgExpctd = imgExpctd + ipermute(subImgInZ,[3,1,2]);
            end

            % Instantiation of target class
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFiltersInXY',synthesisFiltersInXY,...
                'SynthesisFiltersInZ',synthesisFiltersInZ);
            
            % Actual values
            imgActual = ...
                testCase.synthesizer.step(coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepLevel2(testCase,...
                nsubrows,nsubcols,nsublays,ndecsX,ndecsY,ndecsZ,pordXY,pordZ)
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nLevelsXY = 2;
            nDecs = [ ndecsY ndecsX ndecsZ ];
            redundancy_ = 2;
            
            % Filters in XY
            nChsXY = redundancy_*ndecsY*ndecsX;
            lenY = (pordXY+1)*ndecsY;
            lenX = (pordXY+1)*ndecsX;
            synthesisFiltersInXY = zeros(lenY,lenX,nChsXY);
            for iChXY = 1:nChsXY
                synthesisFiltersInXY(:,:,iChXY) = randn(lenY,lenX);
            end
            % Filters in Z
            nChsZ = ndecsZ;
            lenZ = (pordZ+1)*ndecsZ;                
            synthesisFiltersInZ = zeros(lenZ,nChsZ);
            for iChInZ = 1:nChsZ
                synthesisFiltersInZ(:,iChInZ) = randn(lenZ,1);
            end            
            
            % Expected values
            import saivdr.dictionary.generalfb.*      
            %
            subCoefs = [];
            scales = [];
            for iChZ = 1:nChsZ
                subScales = [nsubrows, nsubcols, nsublays];
                for iLvXY = nLevelsXY:-1:1
                    for iChXY = 1:nChsXY
                        if (iChXY ~= 1 || iLvXY == nLevelsXY)
                            subCoefs{iChXY,iLvXY,iChZ} = ...
                                randn(subScales);
                            scales = cat(1,scales,subScales);
                        end
                    end
                    subScales = subScales.*[ ndecsY ndecsX 1 ]; 
                end
            end
            % Coefs.& scales
            nSubbands = nChsZ*( nChsXY + (nLevelsXY-1)*(nChsXY-1) );
            iSubband = 1;
            coefsZ = cell(nSubbands,1);
            for iChZ = 1:nChsZ
                for iChXY = 1:nChsXY
                    coefsZ{iSubband} = subCoefs{iChXY,nLevelsXY,iChZ};
                    iSubband = iSubband + 1;
                end                
                for iLvXY = nLevelsXY-1:-1:1
                    for iChXY = 2:nChsXY
                        coefsZ{iSubband} = subCoefs{iChXY,iLvXY,iChZ};
                        iSubband = iSubband + 1;
                    end
                end
            end
            coefs = cell2mat(...
                cellfun(@(x) x(:),coefsZ,'UniformOutput',false)).';
            %
            phase = 1-mod(nDecs,2); % for phase adjustment required experimentaly
            %
            imgExpctd = 0;
            for iChZ = 1:nChsZ
                subImgInZ = 0;
                iLvXY = nLevelsXY;
                % Interpolation in XY
                for iChXY = 1:nChsXY
                    subImgInXY = subCoefs{iChXY,iLvXY,iChZ};
                    fxy = synthesisFiltersInXY(:,:,iChXY);
                    % Upsample in Y
                    if nDecs(Direction.VERTICAL) > 1 && ...
                            size(subImgInXY,1) == 1
                        tmpImg = cat(1,subImgInXY,...
                            zeros(...
                            nDecs(Direction.VERTICAL)-1,...
                            size(subImgInXY,2),...
                            size(subImgInXY,3)));
                        tmpImg = circshift(tmpImg,phase.*[1 0 0]);
                    else
                        tmpImg = upsample(subImgInXY,...
                            nDecs(Direction.VERTICAL),...
                            phase(Direction.VERTICAL));
                    end
                    % Upsample in X
                    if nDecs(Direction.HORIZONTAL) > 1 && ...
                            size(tmpImg,2) == 1
                        tmpImg = cat(2,tmpImg,...
                            zeros(...
                            size(tmpImg,1),...
                            nDecs(Direction.HORIZONTAL)-1,...
                            size(tmpImg,3)));
                        tmpImg = circshift(tmpImg,phase.*[0 1 0]);
                    else
                        tmpImg = ipermute(upsample(permute(tmpImg,[2,1,3]),...
                            nDecs(Direction.HORIZONTAL),...
                            phase(Direction.HORIZONTAL)),[2,1,3]);
                    end
                    % Filter in XY
                    subImgInZ = subImgInZ ...
                        + imfilter(tmpImg,fxy,'circ','conv');
                end
                iLvXY = iLvXY - 1;
                for iChXY = 1:nChsXY
                    if iChXY == 1
                        subImgInXY = subImgInZ;
                        subImgInZ = 0;
                    else
                        subImgInXY = subCoefs{iChXY,iLvXY,iChZ};
                    end
                    fxy = synthesisFiltersInXY(:,:,iChXY);
                     % Upsample in Y
                    if nDecs(Direction.VERTICAL) > 1 && ...
                            size(subImgInXY,1) == 1
                        tmpImg = cat(1,subImgInXY,...
                            zeros(...
                            nDecs(Direction.VERTICAL)-1,...
                            size(subImgInXY,2),...
                            size(subImgInXY,3)));
                        tmpImg = circshift(tmpImg,phase.*[1 0 0]);
                    else
                        tmpImg = upsample(subImgInXY,...
                            nDecs(Direction.VERTICAL),...
                            phase(Direction.VERTICAL));
                    end
                    % Upsample in X
                    if nDecs(Direction.HORIZONTAL) > 1 && ...
                            size(tmpImg,2) == 1
                        tmpImg = cat(2,tmpImg,...
                            zeros(...
                            size(tmpImg,1),...
                            nDecs(Direction.HORIZONTAL)-1,...
                            size(tmpImg,3)));
                        tmpImg = circshift(tmpImg,phase.*[0 1 0]);
                    else
                        tmpImg = ipermute(upsample(permute(tmpImg,[2,1,3]),...
                            nDecs(Direction.HORIZONTAL),...
                            phase(Direction.HORIZONTAL)),[2,1,3]);
                    end
                    % Filter in XY
                    subImgInZ = subImgInZ ...
                        + imfilter(tmpImg,fxy,'circ','conv');
                end
                fz = synthesisFiltersInZ(:,iChZ);
                % Interpolation in Z
                if nDecs(Direction.DEPTH) > 1 && ...
                        size(subImgInZ,3)==1
                    % Upsample in Z and permute
                    tmpImg = cat(3,subImgInZ,...
                        zeros(...
                        size(subImgInZ,1),...
                        size(subImgInZ,2),...
                        nDecs(Direction.DEPTH)-1));
                    tmpImg = circshift(tmpImg,phase.*[0 0 1]);
                    tmpImg = permute(tmpImg,[3,1,2]);
                else
                    % Upsample in Z and permute
                    tmpImg = upsample(permute(subImgInZ,[3,1,2]),...
                        nDecs(Direction.DEPTH),phase(Direction.DEPTH));
                end
                imgExpctd = imgExpctd ...
                    + ipermute(imfilter(tmpImg,fz,'circ','conv'),[3,1,2]);
            end
            
            % Instantiation of target class
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFiltersInXY',synthesisFiltersInXY,...
                'SynthesisFiltersInZ',synthesisFiltersInZ);
            
            % Actual values
            imgActual = ...
                testCase.synthesizer.step(coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
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
            
            % Filters in XY
            nChsXY = redundancy_*ndecsY*ndecsX;
            lenY = (pordXY_+1)*ndecsY;
            lenX = (pordXY_+1)*ndecsX;
            synthesisFiltersInXY = zeros(lenY,lenX,nChsXY);
            for iChXY = 1:nChsXY
                synthesisFiltersInXY(:,:,iChXY) = randn(lenY,lenX);
            end
            % Filters in Z
            nChsZ = ndecsZ;
            lenZ = (pordZ_+1)*ndecsZ;                
            synthesisFiltersInZ = zeros(lenZ,nChsZ);
            for iChInZ = 1:nChsZ
                synthesisFiltersInZ(:,iChInZ) = randn(lenZ,1);
            end            
            
            % Expected values
            import saivdr.dictionary.generalfb.*      
            %
            subCoefs = [];
            scales = [];
            for iChZ = 1:nChsZ
                subScales = [nsubrows_, nsubcols_, nsublays_];
                for iLvXY = nLevelsXY:-1:1
                    for iChXY = 1:nChsXY
                        if (iChXY ~= 1 || iLvXY == nLevelsXY)
                            subCoefs{iChXY,iLvXY,iChZ} = ...
                                randn(subScales);
                            scales = cat(1,scales,subScales);
                        end
                    end
                    subScales = subScales.*[ ndecsY ndecsX 1 ]; 
                end
            end
            % Coefs.& scales
            nSubbands = nChsZ*( nChsXY + (nLevelsXY-1)*(nChsXY-1) );
            iSubband = 1;
            coefsZ = cell(nSubbands,1);
            for iChZ = 1:nChsZ
                for iChXY = 1:nChsXY
                    coefsZ{iSubband} = subCoefs{iChXY,nLevelsXY,iChZ};
                    iSubband = iSubband + 1;
                end                
                for iLvXY = nLevelsXY-1:-1:1
                    for iChXY = 2:nChsXY
                        coefsZ{iSubband} = subCoefs{iChXY,iLvXY,iChZ};
                        iSubband = iSubband + 1;
                    end
                end
            end
            coefs = cell2mat(...
                cellfun(@(x) x(:),coefsZ,'UniformOutput',false)).';
            %
            phase = 1-mod(nDecs,2); % for phase adjustment required experimentaly
            %
            imgExpctd = 0;
            for iChZ = 1:nChsZ
                subImgInZ = 0;
                for iLvXY = nLevelsXY:-1:1
                    % Interpolation in XY
                    for iChXY = 1:nChsXY
                        if iChXY == 1 && iLvXY ~= nLevelsXY
                            subImgInXY = subImgInZ;
                            subImgInZ = 0;
                        else
                            subImgInXY = subCoefs{iChXY,iLvXY,iChZ};
                        end
                        fxy = synthesisFiltersInXY(:,:,iChXY);
                        % Upsample in Y
                        if nDecs(Direction.VERTICAL) > 1 && ...
                                size(subImgInXY,1) == 1
                            tmpImg = cat(1,subImgInXY,...
                                zeros(...
                                nDecs(Direction.VERTICAL)-1,...
                                size(subImgInXY,2),...
                                size(subImgInXY,3)));
                            tmpImg = circshift(tmpImg,phase.*[1 0 0]);
                        else
                            tmpImg = upsample(subImgInXY,...
                                nDecs(Direction.VERTICAL),...
                                phase(Direction.VERTICAL));
                        end
                        % Upsample in X
                        if nDecs(Direction.HORIZONTAL) > 1 && ...
                                size(tmpImg,2) == 1
                            tmpImg = cat(2,tmpImg,...
                                zeros(...
                                size(tmpImg,1),...
                                nDecs(Direction.HORIZONTAL)-1,...
                                size(tmpImg,3)));
                            tmpImg = circshift(tmpImg,phase.*[0 1 0]);
                        else
                            tmpImg = ipermute(upsample(permute(tmpImg,[2,1,3]),...
                                nDecs(Direction.HORIZONTAL),...
                                phase(Direction.HORIZONTAL)),[2,1,3]);
                        end
                        % Filter in XY
                        subImgInZ = subImgInZ ...
                            + imfilter(tmpImg,fxy,'circ','conv');
                    end
                end
                fz = synthesisFiltersInZ(:,iChZ);
                % Interpolation in Z
                if nDecs(Direction.DEPTH) > 1 && ...
                        size(subImgInZ,3)==1
                    % Upsample in Z and permute
                    tmpImg = cat(3,subImgInZ,...
                        zeros(...
                        size(subImgInZ,1),...
                        size(subImgInZ,2),...
                        nDecs(Direction.DEPTH)-1));
                    tmpImg = circshift(tmpImg,phase.*[0 0 1]);
                    tmpImg = permute(tmpImg,[3,1,2]);
                else
                    % Upsample in Z and permute
                    tmpImg = upsample(permute(subImgInZ,[3,1,2]),...
                        nDecs(Direction.DEPTH),phase(Direction.DEPTH));
                end
                imgExpctd = imgExpctd ...
                    + ipermute(imfilter(tmpImg,fz,'circ','conv'),[3,1,2]);
            end
            
            % Instantiation of target class
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFiltersInXY',synthesisFiltersInXY,...
                'SynthesisFiltersInZ',synthesisFiltersInZ);
            
            % Actual values
            imgActual = ...
                testCase.synthesizer.step(coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
   
        %{
        % Test
        function testStepDec222Ch54Ord000Level1(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            nDecs = [ 2 2 1 ];
            synthesisFilters(:,:,1) = randn(2,2);
            synthesisFilters(:,:,2) = randn(2,2);
            synthesisFilters(:,:,3) = randn(2,2);
            synthesisFilters(:,:,4) = randn(2,2);
            synthesisFilters(:,:,5) = randn(2,2);
            synthesisFilters(:,:,6) = randn(2,2);
            synthesisFilters(:,:,7) = randn(2,2);
            synthesisFilters(:,:,8) = randn(2,2);            
            synthesisFilters(:,:,9) = randn(2,2);            
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(prod(nDecs),3);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX,depth/decZ);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width,depth);
            upsample3_ = @(x,d,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);
            phase = [1 1 0]; % for phase adjustment required experimentaly
            subbandDctImg = zeros(height,width,depth);
            dctImg = zeros(height,width,depth);
            % X-Y filtering
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                updImg = upsample3_(subCoefs{iCh},nDecs,phase);
                for iLay = 1:depth
                    subbandDctImg(:,:,iLay) = imfilter(updImg(:,:,iLay),f,...
                        'conv','circ');
                end
                dctImg = dctImg + subbandDctImg;
            end
            % IDCT for Z direction
            for iCol = 1:width
                for iRow = 1:height
                    imgExpctd(iRow,iCol,:) = idct(dctImg(iRow,iCol,:));
                end
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'SynthesisFilters',synthesisFilters);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch54Ord222Level1(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            nDecs = [ 2 2 1 ];
            synthesisFilters(:,:,1) = randn(6,6);
            synthesisFilters(:,:,2) = randn(6,6);
            synthesisFilters(:,:,3) = randn(6,6);
            synthesisFilters(:,:,4) = randn(6,6);
            synthesisFilters(:,:,5) = randn(6,6);
            synthesisFilters(:,:,6) = randn(6,6);
            synthesisFilters(:,:,7) = randn(6,6);
            synthesisFilters(:,:,8) = randn(6,6);
            synthesisFilters(:,:,9) = randn(6,6);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(prod(nDecs),3);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX,depth/decZ);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width,depth);
            upsample3_ = @(x,d,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);
            phase = [1 1 0]; % for phase adjustment required experimentaly
            subbandDctImg = zeros(height,width,depth);
            dctImg = zeros(height,width,depth);
            % X-Y filtering
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                updImg = upsample3_(subCoefs{iCh},nDecs,phase);
                for iLay = 1:depth
                    subbandDctImg(:,:,iLay) = imfilter(updImg(:,:,iLay),f,...
                        'conv','circ');
                end
                dctImg = dctImg + subbandDctImg;
            end
            % IDCT for Z direction
            for iCol = 1:width
                for iRow = 1:height
                    imgExpctd(iRow,iCol,:) = idct(dctImg(iRow,iCol,:));
                end
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'SynthesisFilters',synthesisFilters);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end        
        
        % Test
        function testStepDec111Ch54Ord111Level1(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            nDecs = [ 1 1 1 ];
            synthesisFilters(:,:,1) = randn(2,2);
            synthesisFilters(:,:,2) = randn(2,2);
            synthesisFilters(:,:,3) = randn(2,2);
            synthesisFilters(:,:,4) = randn(2,2);
            synthesisFilters(:,:,5) = randn(2,2);
            synthesisFilters(:,:,6) = randn(2,2);
            synthesisFilters(:,:,7) = randn(2,2);
            synthesisFilters(:,:,8) = randn(2,2);
            synthesisFilters(:,:,9) = randn(2,2);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(prod(nDecs),3);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX,depth/decZ);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width,depth);
            upsample3_ = @(x,d,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);
            phase = [0 0 0]; % for phase adjustment required experimentaly
            subbandDctImg = zeros(height,width,depth);
            dctImg = zeros(height,width,depth);
            % X-Y filtering
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                updImg = upsample3_(subCoefs{iCh},nDecs,phase);
                for iLay = 1:depth
                    subbandDctImg(:,:,iLay) = imfilter(updImg(:,:,iLay),f,...
                        'conv','circ');
                end
                dctImg = dctImg + subbandDctImg;
            end
            % IDCT for Z direction
            for iCol = 1:width
                for iRow = 1:height
                    imgExpctd(iRow,iCol,:) = idct(dctImg(iRow,iCol,:));
                end
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec321Ch44Ord222Level1(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            nDecs = [ 3 2 1 ];
            synthesisFilters(:,:,1) = randn(9,6);
            synthesisFilters(:,:,2) = randn(9,6);
            synthesisFilters(:,:,3) = randn(9,6);
            synthesisFilters(:,:,4) = randn(9,6);
            synthesisFilters(:,:,5) = randn(9,6);
            synthesisFilters(:,:,6) = randn(9,6);
            synthesisFilters(:,:,7) = randn(9,6);
            synthesisFilters(:,:,8) = randn(9,6);
            synthesisFilters(:,:,9) = randn(9,6);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(prod(nDecs),3);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX,depth/decZ);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width,depth);
            upsample3_ = @(x,d,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);
            phase = [0 1 0]; % for phase adjustment required experimentaly
            subbandDctImg = zeros(height,width,depth);
            dctImg = zeros(height,width,depth);
            % X-Y filtering
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                updImg = upsample3_(subCoefs{iCh},nDecs,phase);
                for iLay = 1:depth
                    subbandDctImg(:,:,iLay) = imfilter(updImg(:,:,iLay),f,...
                        'conv','circ');
                end
                dctImg = dctImg + subbandDctImg;
            end
            % IDCT for Z direction
            for iCol = 1:width
                for iRow = 1:height
                    imgExpctd(iRow,iCol,:) = idct(dctImg(iRow,iCol,:));
                end
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end

        % Test
        function testStepDec222Ch44Ord222Level2(testCase)
            
            % Parameters
            height = 48;
            width  = 64;
            depth  = 32;
            nDecs  = [ 2 2 1 ];
            synthesisFilters(:,:,1) = randn(6,6);
            synthesisFilters(:,:,2) = randn(6,6);
            synthesisFilters(:,:,3) = randn(6,6);
            synthesisFilters(:,:,4) = randn(6,6);
            synthesisFilters(:,:,5) = randn(6,6);
            synthesisFilters(:,:,6) = randn(6,6);
            synthesisFilters(:,:,7) = randn(6,6);
            synthesisFilters(:,:,8) = randn(6,6);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{2} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{3} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{4} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{5} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{6} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{7} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{8} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));            
            subCoefs{9} = rand(height/(decY),width/(decX),depth/(decZ));
            subCoefs{10} = rand(height/(decY),width/(decX),depth/(decZ));
            subCoefs{11} = rand(height/(decY),width/(decX),depth/(decZ));
            subCoefs{12} = rand(height/(decY),width/(decX),depth/(decZ));
            subCoefs{13} = rand(height/(decY),width/(decX),depth/(decZ));
            subCoefs{14} = rand(height/(decY),width/(decX),depth/(decZ));            
            subCoefs{15} = rand(height/(decY),width/(decX),depth/(decZ));                        
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            upsample3_ = @(x,d,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);            
            phase = [ 1 1 0 ]; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                % X-Y filtering
                f = synthesisFilters(:,:,1);
                updImg = upsample3_(subsubCoefs{1},nDecs,phase);
                subbandDctImg = zeros(size(updImg));
                imgExpctd = zeros(size(updImg));
                for iLay = 1:size(updImg,3)
                    subbandDctImg(:,:,iLay) = imfilter(updImg(:,:,iLay),f,...
                        'conv','circ');
                end
                dctImg = subbandDctImg;
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    updImg = upsample3_(subCoefs{iSubband},nDecs,phase);
                    for iLay = 1:size(updImg,3)
                        subbandDctImg(:,:,iLay) = imfilter(updImg(:,:,iLay),f,...
                            'conv','circ');
                    end
                    dctImg = dctImg + subbandDctImg;
                end
                % IDCT for Z direction
                for iCol = 1:size(updImg,2)
                    for iRow = 1:size(updImg,1)
                        imgExpctd(iRow,iCol,:) = idct(dctImg(iRow,iCol,:));
                    end
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'SynthesisFilters',synthesisFilters);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord221Level3(testCase)
            
            % Parameters
            height = 48;
            width  = 64;
            depth  = 32;
            nDecs  = [ 2 2 1 ];
            synthesisFilters(:,:,1) = randn(6,6);
            synthesisFilters(:,:,2) = randn(6,6);
            synthesisFilters(:,:,3) = randn(6,6);
            synthesisFilters(:,:,4) = randn(6,6);
            synthesisFilters(:,:,5) = randn(6,6);
            synthesisFilters(:,:,6) = randn(6,6);
            synthesisFilters(:,:,7) = randn(6,6);
            synthesisFilters(:,:,8) = randn(6,6);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
            subCoefs{2} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
            subCoefs{3} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
            subCoefs{4} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
            subCoefs{5} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
            subCoefs{6} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
            subCoefs{7} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
            subCoefs{8} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
            subCoefs{9} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{10} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{11} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{12} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{13} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{14} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{15} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            subCoefs{16} = rand(height/(decY),width/(decX),depth/(decZ));                        
            subCoefs{17} = rand(height/(decY),width/(decX),depth/(decZ));
            subCoefs{18} = rand(height/(decY),width/(decX),depth/(decZ));
            subCoefs{19} = rand(height/(decY),width/(decX),depth/(decZ));
            subCoefs{20} = rand(height/(decY),width/(decX),depth/(decZ));
            subCoefs{21} = rand(height/(decY),width/(decX),depth/(decZ));
            subCoefs{22} = rand(height/(decY),width/(decX),depth/(decZ));            
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            upsample3_ = @(x,d,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);            
            phase = [ 1 1 0 ]; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                % X-Y filtering
                f = synthesisFilters(:,:,1);
                updImg = upsample3_(subsubCoefs{1},nDecs,phase);
                subbandDctImg = zeros(size(updImg));
                imgExpctd = zeros(size(updImg));
                for iLay = 1:size(updImg,3)
                    subbandDctImg(:,:,iLay) = imfilter(updImg(:,:,iLay),f,...
                        'conv','circ');
                end
                dctImg = subbandDctImg;
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    updImg = upsample3_(subCoefs{iSubband},nDecs,phase);
                    for iLay = 1:size(updImg,3)
                        subbandDctImg(:,:,iLay) = imfilter(updImg(:,:,iLay),f,...
                            'conv','circ');
                    end
                    dctImg = dctImg + subbandDctImg;
                end
                % IDCT for Z direction
                for iCol = 1:size(updImg,2)
                    for iRow = 1:size(updImg,1)
                        imgExpctd(iRow,iCol,:) = idct(dctImg(iRow,iCol,:));
                    end
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'SynthesisFilters',synthesisFilters);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        %}
        %{
%         % Test
%         function testStepDec222Ch44Ord000Level1Freq(testCase)
%             
%             % Parameters
%             height = 48;
%             width = 64;
%             depth = 32;
%             nDecs = [ 2 2 2 ];
%             synthesisFilters(:,:,:,1) = randn(2,2,2);
%             synthesisFilters(:,:,:,2) = randn(2,2,2);
%             synthesisFilters(:,:,:,3) = randn(2,2,2);
%             synthesisFilters(:,:,:,4) = randn(2,2,2);
%             synthesisFilters(:,:,:,5) = randn(2,2,2);
%             synthesisFilters(:,:,:,6) = randn(2,2,2);
%             synthesisFilters(:,:,:,7) = randn(2,2,2);
%             synthesisFilters(:,:,:,8) = randn(2,2,2);            
%             %nLevels = 1;
%             
%             % Expected values
%             import saivdr.dictionary.utility.Direction
%             decY = nDecs(Direction.VERTICAL);
%             decX = nDecs(Direction.HORIZONTAL);
%             decZ = nDecs(Direction.DEPTH);
%             nChs = size(synthesisFilters,4);
%             subCoefs = cell(nChs,1);
%             coefs = zeros(1,height*width*depth);
%             scales = zeros(prod(nDecs),3);
%             sIdx = 1;
%             for iCh = 1:nChs
%                 subImg = rand(height/decY,width/decX,depth/decZ);
%                 subCoefs{iCh} = subImg;
%                 eIdx = sIdx + numel(subImg) - 1;
%                 coefs(sIdx:eIdx) = subImg(:).';
%                 scales(iCh,:) = size(subImg);
%                 sIdx = eIdx + 1;
%             end
%             imgExpctd = zeros(height,width,depth);
%             upsample3_ = @(x,d,p) ...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);
%             phase = [1 1 1]; % for phase adjustment required experimentaly
%             for iCh = 1:nChs
%                 f = synthesisFilters(:,:,:,iCh);
%                 subbandImg = imfilter(...
%                     upsample3_(subCoefs{iCh},nDecs,phase),...
%                     f,'conv','circ');
%                 imgExpctd = imgExpctd + subbandImg;
%             end
%             
%             % Instantiation of target class
%             import saivdr.dictionary.generalfb.*
%             testCase.synthesizer = Synthesis3dSystem(...
%                 'SynthesisFilters',synthesisFilters,...
%                 'FilterDomain','Frequency');
%             
%             % Actual values
%             imgActual = ...
%                 step(testCase.synthesizer,coefs,scales);
%             
%             % Evaluation
%             testCase.verifySize(imgActual,size(imgExpctd),...
%                 'Actual image size is different from the expected one.');
%             diff = max(abs(imgExpctd(:) - imgActual(:)));
%             testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
%         end
%         
%         % Test
%         function testStepDec222Ch54Ord000Level1Freq(testCase)
%             
%             % Parameters
%             height = 48;
%             width = 64;
%             depth = 32;
%             nDecs = [ 2 2 2 ];
%             synthesisFilters(:,:,:,1) = randn(2,2,2);
%             synthesisFilters(:,:,:,2) = randn(2,2,2);
%             synthesisFilters(:,:,:,3) = randn(2,2,2);
%             synthesisFilters(:,:,:,4) = randn(2,2,2);
%             synthesisFilters(:,:,:,5) = randn(2,2,2);
%             synthesisFilters(:,:,:,6) = randn(2,2,2);
%             synthesisFilters(:,:,:,7) = randn(2,2,2);
%             synthesisFilters(:,:,:,8) = randn(2,2,2);            
%             synthesisFilters(:,:,:,9) = randn(2,2,2);            
%             %nLevels = 1;
%             
%             % Expected values
%             import saivdr.dictionary.utility.Direction
%             decY = nDecs(Direction.VERTICAL);
%             decX = nDecs(Direction.HORIZONTAL);
%             decZ = nDecs(Direction.DEPTH);
%             nChs = size(synthesisFilters,4);
%             subCoefs = cell(nChs,1);
%             coefs = zeros(1,height*width*depth);
%             scales = zeros(prod(nDecs),3);
%             sIdx = 1;
%             for iCh = 1:nChs
%                 subImg = rand(height/decY,width/decX,depth/decZ);
%                 subCoefs{iCh} = subImg;
%                 eIdx = sIdx + numel(subImg) - 1;
%                 coefs(sIdx:eIdx) = subImg(:).';
%                 scales(iCh,:) = size(subImg);
%                 sIdx = eIdx + 1;
%             end
%             imgExpctd = zeros(height,width,depth);
%             upsample3_ = @(x,d,p) ...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);
%             phase = [1 1 1]; % for phase adjustment required experimentaly
%             for iCh = 1:nChs
%                 f = synthesisFilters(:,:,:,iCh);
%                 subbandImg = imfilter(...
%                     upsample3_(subCoefs{iCh},nDecs,phase),...
%                     f,'conv','circ');
%                 imgExpctd = imgExpctd + subbandImg;
%             end
%             
%             % Instantiation of target class
%             import saivdr.dictionary.generalfb.*
%             testCase.synthesizer = Synthesis3dSystem(...
%                 'SynthesisFilters',synthesisFilters,...
%                 'FilterDomain','Frequency');
%             
%             % Actual values
%             imgActual = ...
%                 step(testCase.synthesizer,coefs,scales);
%             
%             % Evaluation
%             testCase.verifySize(imgActual,size(imgExpctd),...
%                 'Actual image size is different from the expected one.');
%             diff = max(abs(imgExpctd(:) - imgActual(:)));
%             testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
%         end
%         
%         % Test
%         function testStepDec222Ch54Ord222Level1Freq(testCase)
%             
%             % Parameters
%             height = 48;
%             width = 64;
%             depth = 32;
%             nDecs = [ 2 2 2 ];
%             synthesisFilters(:,:,:,1) = randn(6,6,6);
%             synthesisFilters(:,:,:,2) = randn(6,6,6);
%             synthesisFilters(:,:,:,3) = randn(6,6,6);
%             synthesisFilters(:,:,:,4) = randn(6,6,6);
%             synthesisFilters(:,:,:,5) = randn(6,6,6);
%             synthesisFilters(:,:,:,6) = randn(6,6,6);
%             synthesisFilters(:,:,:,7) = randn(6,6,6);
%             synthesisFilters(:,:,:,8) = randn(6,6,6);
%             synthesisFilters(:,:,:,9) = randn(6,6,6);
%             %nLevels = 1;
%             
%             % Expected values
%             import saivdr.dictionary.utility.Direction
%             decY = nDecs(Direction.VERTICAL);
%             decX = nDecs(Direction.HORIZONTAL);
%             decZ = nDecs(Direction.DEPTH);
%             nChs = size(synthesisFilters,4);
%             subCoefs = cell(nChs,1);
%             coefs = zeros(1,height*width*depth);
%             scales = zeros(prod(nDecs),3);
%             sIdx = 1;
%             for iCh = 1:nChs
%                 subImg = rand(height/decY,width/decX,depth/decZ);
%                 subCoefs{iCh} = subImg;
%                 eIdx = sIdx + numel(subImg) - 1;
%                 coefs(sIdx:eIdx) = subImg(:).';
%                 scales(iCh,:) = size(subImg);
%                 sIdx = eIdx + 1;
%             end
%             imgExpctd = zeros(height,width,depth);
%             upsample3_ = @(x,d,p) ...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);
%             phase = [1 1 1]; % for phase adjustment required experimentaly
%             for iCh = 1:nChs
%                 f = synthesisFilters(:,:,:,iCh);
%                 subbandImg = imfilter(...
%                     upsample3_(subCoefs{iCh},nDecs,phase),...
%                     f,'conv','circ');
%                 imgExpctd = imgExpctd + subbandImg;
%             end
%             
%             % Instantiation of target class
%             import saivdr.dictionary.generalfb.*
%             testCase.synthesizer = Synthesis3dSystem(...
%                 'SynthesisFilters',synthesisFilters,...
%                 'FilterDomain','Frequency');
%             
%             % Actual values
%             imgActual = ...
%                 step(testCase.synthesizer,coefs,scales);
%             
%             % Evaluation
%             testCase.verifySize(imgActual,size(imgExpctd),...
%                 'Actual image size is different from the expected one.');
%             diff = max(abs(imgExpctd(:) - imgActual(:)));
%             testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
%         end        
%         
%         % Test
%         function testStepDec111Ch54Ord111Level1Freq(testCase)
%             
%             % Parameters
%             height = 48;
%             width = 64;
%             depth = 32;
%             nDecs = [ 1 1 1 ];
%             synthesisFilters(:,:,:,1) = randn(2,2,2);
%             synthesisFilters(:,:,:,2) = randn(2,2,2);
%             synthesisFilters(:,:,:,3) = randn(2,2,2);
%             synthesisFilters(:,:,:,4) = randn(2,2,2);
%             synthesisFilters(:,:,:,5) = randn(2,2,2);
%             synthesisFilters(:,:,:,6) = randn(2,2,2);
%             synthesisFilters(:,:,:,7) = randn(2,2,2);
%             synthesisFilters(:,:,:,8) = randn(2,2,2);
%             synthesisFilters(:,:,:,9) = randn(2,2,2);
%             %nLevels = 1;
%             
%             % Expected values
%             import saivdr.dictionary.utility.Direction
%             decY = nDecs(Direction.VERTICAL);
%             decX = nDecs(Direction.HORIZONTAL);
%             decZ = nDecs(Direction.DEPTH);
%             nChs = size(synthesisFilters,4);
%             subCoefs = cell(nChs,1);
%             coefs = zeros(1,height*width*depth);
%             scales = zeros(prod(nDecs),3);
%             sIdx = 1;
%             for iCh = 1:nChs
%                 subImg = rand(height/decY,width/decX,depth/decZ);
%                 subCoefs{iCh} = subImg;
%                 eIdx = sIdx + numel(subImg) - 1;
%                 coefs(sIdx:eIdx) = subImg(:).';
%                 scales(iCh,:) = size(subImg);
%                 sIdx = eIdx + 1;
%             end
%             imgExpctd = zeros(height,width,depth);
%             upsample3_ = @(x,d,p) ...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);
%             phase = [0 0 0]; % for phase adjustment required experimentaly
%             for iCh = 1:nChs
%                 f = synthesisFilters(:,:,:,iCh);
%                 subbandImg = imfilter(...
%                     upsample3_(subCoefs{iCh},nDecs,phase),...
%                     f,'conv','circ');
%                 imgExpctd = imgExpctd + subbandImg;
%             end
%             
%             % Instantiation of target class
%             import saivdr.dictionary.generalfb.*
%             testCase.synthesizer = Synthesis3dSystem(...
%                 'DecimationFactor',nDecs,...
%                 'SynthesisFilters',synthesisFilters,...
%                 'FilterDomain','Frequency');
%             
%             % Actual values
%             imgActual = ...
%                 step(testCase.synthesizer,coefs,scales);
%             
%             % Evaluation
%             testCase.verifySize(imgActual,size(imgExpctd),...
%                 'Actual image size is different from the expected one.');
%             diff = max(abs(imgExpctd(:) - imgActual(:)));
%             testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
%         end
%         
%         % Test
%         function testStepDec321Ch44Ord222Level1Freq(testCase)
%             
%             % Parameters
%             height = 48;
%             width = 64;
%             depth = 32;
%             nDecs = [ 3 2 1 ];
%             synthesisFilters(:,:,:,1) = randn(9,6,3);
%             synthesisFilters(:,:,:,2) = randn(9,6,3);
%             synthesisFilters(:,:,:,3) = randn(9,6,3);
%             synthesisFilters(:,:,:,4) = randn(9,6,3);
%             synthesisFilters(:,:,:,5) = randn(9,6,3);
%             synthesisFilters(:,:,:,6) = randn(9,6,3);
%             synthesisFilters(:,:,:,7) = randn(9,6,3);
%             synthesisFilters(:,:,:,8) = randn(9,6,3);
%             synthesisFilters(:,:,:,9) = randn(9,6,3);
%             %nLevels = 1;
%             
%             % Expected values
%             import saivdr.dictionary.utility.Direction
%             decY = nDecs(Direction.VERTICAL);
%             decX = nDecs(Direction.HORIZONTAL);
%             decZ = nDecs(Direction.DEPTH);
%             nChs = size(synthesisFilters,4);
%             subCoefs = cell(nChs,1);
%             coefs = zeros(1,height*width*depth);
%             scales = zeros(prod(nDecs),3);
%             sIdx = 1;
%             for iCh = 1:nChs
%                 subImg = rand(height/decY,width/decX,depth/decZ);
%                 subCoefs{iCh} = subImg;
%                 eIdx = sIdx + numel(subImg) - 1;
%                 coefs(sIdx:eIdx) = subImg(:).';
%                 scales(iCh,:) = size(subImg);
%                 sIdx = eIdx + 1;
%             end
%             imgExpctd = zeros(height,width,depth);
%             upsample3_ = @(x,d,p) ...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);
%             phase = [0 1 0]; % for phase adjustment required experimentaly
%             for iCh = 1:nChs
%                 f = synthesisFilters(:,:,:,iCh);
%                 subbandImg = imfilter(...
%                     upsample3_(subCoefs{iCh},nDecs,phase),...
%                     f,'conv','circ');
%                 imgExpctd = imgExpctd + subbandImg;
%             end
%             
%             % Instantiation of target class
%             import saivdr.dictionary.generalfb.*
%             testCase.synthesizer = Synthesis3dSystem(...
%                 'DecimationFactor',nDecs,...
%                 'SynthesisFilters',synthesisFilters,...
%                 'FilterDomain','Frequency');
%             
%             % Actual values
%             imgActual = ...
%                 step(testCase.synthesizer,coefs,scales);
%             
%             % Evaluation
%             testCase.verifySize(imgActual,size(imgExpctd),...
%                 'Actual image size is different from the expected one.');
%             diff = max(abs(imgExpctd(:) - imgActual(:)));
%             testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
%         end
% 
%         % Test
%         function testStepDec222Ch44Ord222Level2Freq(testCase)
%             
%             % Parameters
%             height = 48;
%             width  = 64;
%             depth  = 32;
%             nDecs  = [ 2 2 2 ];
%             synthesisFilters(:,:,:,1) = randn(6,6,6);
%             synthesisFilters(:,:,:,2) = randn(6,6,6);
%             synthesisFilters(:,:,:,3) = randn(6,6,6);
%             synthesisFilters(:,:,:,4) = randn(6,6,6);
%             synthesisFilters(:,:,:,5) = randn(6,6,6);
%             synthesisFilters(:,:,:,6) = randn(6,6,6);
%             synthesisFilters(:,:,:,7) = randn(6,6,6);
%             synthesisFilters(:,:,:,8) = randn(6,6,6);
%             nLevels = 2;
%             
%             % Preparation
%             import saivdr.dictionary.utility.Direction
%             decY = nDecs(Direction.VERTICAL);
%             decX = nDecs(Direction.HORIZONTAL);
%             decZ = nDecs(Direction.DEPTH);
%             nChs = size(synthesisFilters,4);
%             subCoefs = cell(nLevels*(nChs-1)+1,1);
%             subCoefs{1} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{2} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{3} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{4} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{5} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{6} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{7} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{8} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));            
%             subCoefs{9} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{10} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{11} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{12} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{13} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{14} = rand(height/(decY),width/(decX),depth/(decZ));            
%             subCoefs{15} = rand(height/(decY),width/(decX),depth/(decZ));                        
%             nSubbands = length(subCoefs);
%             scales = zeros(nSubbands,3);
%             sIdx = 1;
%             for iSubband = 1:nSubbands
%                 scales(iSubband,:) = size(subCoefs{iSubband});
%                 eIdx = sIdx + prod(scales(iSubband,:))-1;
%                 coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
%                 sIdx = eIdx + 1;
%             end
%             
%             % Expected values
%             upsample3_ = @(x,d,p) ...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);            
%             phase = [ 1 1 1 ]; % for phase adjustment required experimentaly
%             subsubCoefs = cell(nChs,1);
%             subsubCoefs{1} = subCoefs{1};
%             for iLevel = 1:nLevels
%                 f = synthesisFilters(:,:,:,1);
%                 imgExpctd = imfilter(...
%                     upsample3_(subsubCoefs{1},nDecs,phase),...
%                     f,'conv','circ');
%                 for iCh = 2:nChs
%                     f = synthesisFilters(:,:,:,iCh);
%                     iSubband = (iLevel-1)*(nChs-1)+iCh;
%                     subbandImg = imfilter(...
%                         upsample3_(subCoefs{iSubband},nDecs,phase),...
%                         f,'conv','circ');
%                     imgExpctd = imgExpctd + subbandImg;
%                 end
%                 subsubCoefs{1}=imgExpctd;
%             end
%             
%             % Instantiation of target class
%             import saivdr.dictionary.generalfb.*
%             testCase.synthesizer = Synthesis3dSystem(...
%                 'SynthesisFilters',synthesisFilters,...
%                 'FilterDomain','Frequency');
%             
%             % Actual values
%             imgActual = step(testCase.synthesizer,coefs,scales);
%             
%             % Evaluation
%             testCase.verifySize(imgActual,size(imgExpctd),...
%                 'Actual image size is different from the expected one.');
%             diff = max(abs(imgExpctd(:) - imgActual(:)));
%             testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
%         end
%         
%         % Test
%         function testStepDec222Ch44Ord222Level3Freq(testCase)
%             
%             % Parameters
%             height = 48;
%             width  = 64;
%             depth  = 32;
%             nDecs  = [ 2 2 2 ];
%             synthesisFilters(:,:,:,1) = randn(6,6,6);
%             synthesisFilters(:,:,:,2) = randn(6,6,6);
%             synthesisFilters(:,:,:,3) = randn(6,6,6);
%             synthesisFilters(:,:,:,4) = randn(6,6,6);
%             synthesisFilters(:,:,:,5) = randn(6,6,6);
%             synthesisFilters(:,:,:,6) = randn(6,6,6);
%             synthesisFilters(:,:,:,7) = randn(6,6,6);
%             synthesisFilters(:,:,:,8) = randn(6,6,6);
%             nLevels = 3;
%             
%             % Preparation
%             import saivdr.dictionary.utility.Direction
%             decY = nDecs(Direction.VERTICAL);
%             decX = nDecs(Direction.HORIZONTAL);
%             decZ = nDecs(Direction.DEPTH);
%             nChs = size(synthesisFilters,4);
%             subCoefs = cell(nLevels*(nChs-1)+1,1);
%             subCoefs{1} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{2} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{3} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{4} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{5} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{6} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{7} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{8} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{9} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{10} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{11} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{12} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{13} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{14} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{15} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{16} = rand(height/(decY),width/(decX),depth/(decZ));                        
%             subCoefs{17} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{18} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{19} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{20} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{21} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{22} = rand(height/(decY),width/(decX),depth/(decZ));            
%             nSubbands = length(subCoefs);
%             scales = zeros(nSubbands,3);
%             sIdx = 1;
%             for iSubband = 1:nSubbands
%                 scales(iSubband,:) = size(subCoefs{iSubband});
%                 eIdx = sIdx + prod(scales(iSubband,:))-1;
%                 coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
%                 sIdx = eIdx + 1;
%             end
%             
%             % Expected values
%             upsample3_ = @(x,d,p) ...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);            
%             phase = [ 1 1 1 ]; % for phase adjustment required experimentaly
%             subsubCoefs = cell(nChs,1);
%             subsubCoefs{1} = subCoefs{1};
%             for iLevel = 1:nLevels
%                 f = synthesisFilters(:,:,:,1);
%                 imgExpctd = imfilter(...
%                     upsample3_(subsubCoefs{1},nDecs,phase),...
%                     f,'conv','circ');
%                 for iCh = 2:nChs
%                     f = synthesisFilters(:,:,:,iCh);
%                     iSubband = (iLevel-1)*(nChs-1)+iCh;
%                     subbandImg = imfilter(...
%                         upsample3_(subCoefs{iSubband},nDecs,phase),...
%                         f,'conv','circ');
%                     imgExpctd = imgExpctd + subbandImg;
%                 end
%                 subsubCoefs{1}=imgExpctd;
%             end
%             
%             % Instantiation of target class
%             import saivdr.dictionary.generalfb.*
%             testCase.synthesizer = Synthesis3dSystem(...
%                 'SynthesisFilters',synthesisFilters,...
%                 'FilterDomain','Frequency');
%             
%             % Actual values
%             imgActual = step(testCase.synthesizer,coefs,scales);
%             
%             % Evaluation
%             testCase.verifySize(imgActual,size(imgExpctd),...
%                 'Actual image size is different from the expected one.');
%             diff = max(abs(imgExpctd(:) - imgActual(:)));
%             testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
%         end
%}
    %{
        function testStepDec234Ch1414Ord222Level1(testCase)
            
            % Parameters
            height = 8*2;
            width  = 12*3;
            depth  = 16*4;
            nDecs  = [ 2 3 1 ];
            synthesisFilters = zeros(6,9,28);
            for iCh = 1:28
                synthesisFilters(:,:,iCh) = randn(6,9);
            end
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY),width/(decX),depth/(decZ));
            for iCh = 2:28
                subCoefs{iCh} = randn(height/(decY),width/(decX),depth/(decZ));
            end
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            upsample3_ = @(x,d,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);            
            phase = [ 1 0 0 ]; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                % X-Y filtering
                f = synthesisFilters(:,:,1);
                updImg = upsample3_(subsubCoefs{1},nDecs,phase);
                subbandDctImg = zeros(size(updImg));
                imgExpctd = zeros(size(updImg));
                for iLay = 1:size(updImg,3)
                    subbandDctImg(:,:,iLay) = imfilter(updImg(:,:,iLay),f,...
                        'conv','circ');
                end
                dctImg = subbandDctImg;
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    updImg = upsample3_(subCoefs{iSubband},nDecs,phase);
                    for iLay = 1:size(updImg,3)
                        subbandDctImg(:,:,iLay) = imfilter(updImg(:,:,iLay),f,...
                            'conv','circ');
                    end
                    dctImg = dctImg + subbandDctImg;
                end
                % IDCT for Z direction
                for iCol = 1:size(updImg,2)
                    for iRow = 1:size(updImg,1)
                        imgExpctd(iRow,iCol,:) = idct(dctImg(iRow,iCol,:));
                    end
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Spatial');
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        
        % Test
        function testStepDec234Ch1414Ord222Level2(testCase)
            
            % Parameters
            height = 8*2^2;
            width  = 12*3^2;
            depth  = 16*4^2;
            nDecs  = [ 2 3 4 ];
            synthesisFilters = zeros(6,9,28);
            for iCh = 1:28
                synthesisFilters(:,:,iCh) = randn(6,9);
            end
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
            for iCh = 2:28
                subCoefs{iCh} = randn(height/(decY^2),width/(decX^2),depth/(decZ^2));
                subCoefs{iCh+27} = randn(height/(decY),width/(decX),depth/(decZ));
            end
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            upsample3_ = @(x,d,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);            
            phase = [ 1 0 1 ]; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                % X-Y filtering
                f = synthesisFilters(:,:,1);
                updImg = upsample3_(subsubCoefs{1},nDecs,phase);
                subbandDctImg = zeros(size(updImg));
                imgExpctd = zeros(size(updImg));
                for iLay = 1:size(updImg,3)
                    subbandDctImg(:,:,iLay) = imfilter(updImg(:,:,iLay),f,...
                        'conv','circ');
                end
                dctImg = subbandDctImg;
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    updImg = upsample3_(subCoefs{iSubband},nDecs,phase);
                    for iLay = 1:size(updImg,3)
                        subbandDctImg(:,:,iLay) = imfilter(updImg(:,:,iLay),f,...
                            'conv','circ');
                    end
                    dctImg = dctImg + subbandDctImg;
                end
                % IDCT for Z direction
                for iCol = 1:size(updImg,2)
                    for iRow = 1:size(updImg,1)
                        imgExpctd(iRow,iCol,:) = idct(dctImg(iRow,iCol,:));
                    end
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2plus1dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Spatial');
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
%}
        %{
%         %Test
%         function testStepDec234Ch1414Ord222Level2Freq(testCase)
%             
%             % Parameters
%             height = 8*2^2;
%             width  = 12*3^2;
%             depth  = 16*4^2;
%             nDecs  = [ 2 3 4 ];
%             synthesisFilters = zeros(6,9,12,28);
%             for iCh = 1:28
%                 synthesisFilters(:,:,:,iCh) = randn(6,9,12);
%             end
%             nLevels = 2;
%             
%             % Preparation
%             import saivdr.dictionary.utility.Direction
%             decY = nDecs(Direction.VERTICAL);
%             decX = nDecs(Direction.HORIZONTAL);
%             decZ = nDecs(Direction.DEPTH);
%             nChs = size(synthesisFilters,4);
%             subCoefs = cell(nLevels*(nChs-1)+1,1);
%             subCoefs{1} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             for iCh = 2:28
%                 subCoefs{iCh} = randn(height/(decY^2),width/(decX^2),depth/(decZ^2));
%                 subCoefs{iCh+27} = randn(height/(decY),width/(decX),depth/(decZ));
%             end
%             nSubbands = length(subCoefs);
%             scales = zeros(nSubbands,3);
%             sIdx = 1;
%             for iSubband = 1:nSubbands
%                 scales(iSubband,:) = size(subCoefs{iSubband});
%                 eIdx = sIdx + prod(scales(iSubband,:))-1;
%                 coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
%                 sIdx = eIdx + 1;
%             end
%             
%             % Expected values
%             upsample3_ = @(x,d,p) ...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);            
%             phase = [ 1 0 1 ]; % for phase adjustment required experimentaly
%             subsubCoefs = cell(nChs,1);
%             subsubCoefs{1} = subCoefs{1};
%             for iLevel = 1:nLevels
%                 f = synthesisFilters(:,:,:,1);
%                 imgExpctd = imfilter(...
%                     upsample3_(subsubCoefs{1},nDecs,phase),...
%                     f,'conv','circ');
%                 for iCh = 2:nChs
%                     f = synthesisFilters(:,:,:,iCh);
%                     iSubband = (iLevel-1)*(nChs-1)+iCh;
%                     subbandImg = imfilter(...
%                         upsample3_(subCoefs{iSubband},nDecs,phase),...
%                         f,'conv','circ');
%                     imgExpctd = imgExpctd + subbandImg;
%                 end
%                 subsubCoefs{1}=imgExpctd;
%             end
%             
%             % Instantiation of target class
%             import saivdr.dictionary.generalfb.*
%             testCase.synthesizer = Synthesis3dSystem(...
%                 'DecimationFactor',nDecs,...
%                 'SynthesisFilters',synthesisFilters,...
%                 'FilterDomain','Frequency');
%             
%             % Actual values
%             imgActual = step(testCase.synthesizer,coefs,scales);
%             
%             % Evaluation
%             testCase.verifySize(imgActual,size(imgExpctd),...
%                 'Actual image size is different from the expected one.');
%             diff = max(abs(imgExpctd(:) - imgActual(:)));
%             testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
%         end
%         
%     %Test
%         function testStepDec234Ch1414Ord222Level2FreqUseGpuFalse(testCase)
%             
%             % Parameters
%             height = 8*2^2;
%             width  = 12*3^2;
%             depth  = 16*4^2;
%             nDecs  = [ 2 3 4 ];
%             useGpu = false;
%             synthesisFilters = zeros(6,9,12,28);
%             for iCh = 1:28
%                 synthesisFilters(:,:,:,iCh) = randn(6,9,12);
%             end
%             nLevels = 2;
%             
%             % Preparation
%             import saivdr.dictionary.utility.Direction
%             decY = nDecs(Direction.VERTICAL);
%             decX = nDecs(Direction.HORIZONTAL);
%             decZ = nDecs(Direction.DEPTH);
%             nChs = size(synthesisFilters,4);
%             subCoefs = cell(nLevels*(nChs-1)+1,1);
%             subCoefs{1} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             for iCh = 2:28
%                 subCoefs{iCh} = randn(height/(decY^2),width/(decX^2),depth/(decZ^2));
%                 subCoefs{iCh+27} = randn(height/(decY),width/(decX),depth/(decZ));
%             end
%             nSubbands = length(subCoefs);
%             scales = zeros(nSubbands,3);
%             sIdx = 1;
%             for iSubband = 1:nSubbands
%                 scales(iSubband,:) = size(subCoefs{iSubband});
%                 eIdx = sIdx + prod(scales(iSubband,:))-1;
%                 coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
%                 sIdx = eIdx + 1;
%             end
%             
%             % Expected values
%             upsample3_ = @(x,d,p) ...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);            
%             phase = [ 1 0 1 ]; % for phase adjustment required experimentaly
%             subsubCoefs = cell(nChs,1);
%             subsubCoefs{1} = subCoefs{1};
%             for iLevel = 1:nLevels
%                 f = synthesisFilters(:,:,:,1);
%                 imgExpctd = imfilter(...
%                     upsample3_(subsubCoefs{1},nDecs,phase),...
%                     f,'conv','circ');
%                 for iCh = 2:nChs
%                     f = synthesisFilters(:,:,:,iCh);
%                     iSubband = (iLevel-1)*(nChs-1)+iCh;
%                     subbandImg = imfilter(...
%                         upsample3_(subCoefs{iSubband},nDecs,phase),...
%                         f,'conv','circ');
%                     imgExpctd = imgExpctd + subbandImg;
%                 end
%                 subsubCoefs{1}=imgExpctd;
%             end
%             
%             % Instantiation of target class
%             import saivdr.dictionary.generalfb.*
%             testCase.synthesizer = Synthesis3dSystem(...
%                 'DecimationFactor',nDecs,...
%                 'SynthesisFilters',synthesisFilters,...
%                 'FilterDomain','Frequency',...
%                 'UseGpu',useGpu);
%             
%             % Actual values
%             imgActual = step(testCase.synthesizer,coefs,scales);
%             
%             % Evaluation
%             testCase.verifySize(imgActual,size(imgExpctd),...
%                 'Actual image size is different from the expected one.');
%             diff = max(abs(imgExpctd(:) - imgActual(:)));
%             testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
%         end
%         
%         % Test
%         function testClone(testCase)
%             
%             % Parameters
%             height = 48;
%             width  = 64;
%             depth  = 32;
%             nDecs  = [ 2 2 2 ];
%             synthesisFilters(:,:,:,1) = randn(6,6,6);
%             synthesisFilters(:,:,:,2) = randn(6,6,6);
%             synthesisFilters(:,:,:,3) = randn(6,6,6);
%             synthesisFilters(:,:,:,4) = randn(6,6,6);
%             synthesisFilters(:,:,:,5) = randn(6,6,6);
%             synthesisFilters(:,:,:,6) = randn(6,6,6);
%             synthesisFilters(:,:,:,7) = randn(6,6,6);
%             synthesisFilters(:,:,:,8) = randn(6,6,6);
%             nLevels = 3;
%             
%             % Preparation
%             import saivdr.dictionary.utility.Direction
%             decY = nDecs(Direction.VERTICAL);
%             decX = nDecs(Direction.HORIZONTAL);
%             decZ = nDecs(Direction.DEPTH);
%             nChs = size(synthesisFilters,4);
%             subCoefs = cell(nLevels*(nChs-1)+1,1);
%             subCoefs{1} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{2} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{3} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{4} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{5} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{6} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{7} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{8} = rand(height/(decY^3),width/(decX^3),depth/(decZ^3));
%             subCoefs{9} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{10} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{11} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{12} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{13} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{14} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{15} = rand(height/(decY^2),width/(decX^2),depth/(decZ^2));
%             subCoefs{16} = rand(height/(decY),width/(decX),depth/(decZ));                        
%             subCoefs{17} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{18} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{19} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{20} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{21} = rand(height/(decY),width/(decX),depth/(decZ));
%             subCoefs{22} = rand(height/(decY),width/(decX),depth/(decZ));            
%             nSubbands = length(subCoefs);
%             scales = zeros(nSubbands,3);
%             sIdx = 1;
%             for iSubband = 1:nSubbands
%                 scales(iSubband,:) = size(subCoefs{iSubband});
%                 eIdx = sIdx + prod(scales(iSubband,:))-1;
%                 coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
%                 sIdx = eIdx + 1;
%             end
%             
%             % Expected values
%             upsample3_ = @(x,d,p) ...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(...
%                 shiftdim(upsample(x,d(1),p(1)),1),d(2),p(2)),1),d(3),p(3)),1);            
%             phase = [ 1 1 1 ]; % for phase adjustment required experimentaly
%             subsubCoefs = cell(nChs,1);
%             subsubCoefs{1} = subCoefs{1};
%             for iLevel = 1:nLevels
%                 f = synthesisFilters(:,:,:,1);
%                 imgExpctd = imfilter(...
%                     upsample3_(subsubCoefs{1},nDecs,phase),...
%                     f,'conv','circ');
%                 for iCh = 2:nChs
%                     f = synthesisFilters(:,:,:,iCh);
%                     iSubband = (iLevel-1)*(nChs-1)+iCh;
%                     subbandImg = imfilter(...
%                         upsample3_(subCoefs{iSubband},nDecs,phase),...
%                         f,'conv','circ');
%                     imgExpctd = imgExpctd + subbandImg;
%                 end
%                 subsubCoefs{1}=imgExpctd;
%             end
%             
%             % Instantiation of target class
%             import saivdr.dictionary.generalfb.*
%             testCase.synthesizer = Synthesis3dSystem(...
%                 'SynthesisFilters',synthesisFilters,...
%                 'FilterDomain','Frequency');
%             cloneSynthesizer = clone(testCase.synthesizer);
%             
%             % Actual values
%             imgActual = step(cloneSynthesizer,coefs,scales);
%             
%             % Evaluation
%             testCase.verifySize(imgActual,size(imgExpctd),...
%                 'Actual image size is different from the expected one.');
%             diff = max(abs(imgExpctd(:) - imgActual(:)));
%             testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
%         end        
    end
    %}
    end
end

