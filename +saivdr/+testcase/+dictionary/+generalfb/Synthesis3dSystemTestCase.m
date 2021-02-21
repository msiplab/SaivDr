classdef Synthesis3dSystemTestCase < matlab.unittest.TestCase
    %SYNTHESIS3DSYSTEMTESTCASE Test case for Synthesis3dSystem
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015-2017, Shogo MURAMATSU
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
        nsubrows = { 2, 4 };
        nsubcols = { 2, 4 };
        nsublays = { 1, 2 };
        %filterdom = { 'Spatial', 'Frequency' };
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
            synthesisFiltersExpctd = [];
            decimationFactorExpctd = [ 2 2 2 ];
            frmbdExpctd  = [];
            filterDomainExpctd = 'Spatial';
            boundaryOperationExpctd = 'Circular';
            
            % Instantiation
            testCase.synthesizer = Synthesis3dSystem();
            
            % Actual value
            synthesisFiltersActual = get(testCase.synthesizer,'SynthesisFilters');
            decimationFactorActual = get(testCase.synthesizer,'DecimationFactor');
            frmbdActual  = get(testCase.synthesizer,'FrameBound');
            filterDomainActual = get(testCase.synthesizer,'FilterDomain');
            boundaryOperationActual = get(testCase.synthesizer,'BoundaryOperation');  
            
            % Evaluation
            testCase.assertEqual(synthesisFiltersActual,synthesisFiltersExpctd);
            testCase.assertEqual(decimationFactorActual,decimationFactorExpctd);
            testCase.assertEqual(frmbdActual,frmbdExpctd);
            testCase.assertEqual(filterDomainActual,filterDomainExpctd);
            testCase.assertEqual(boundaryOperationActual,boundaryOperationExpctd);            
        end
                
        % Test
        function testSynthesisFilters(testCase)
            
            % Expected values
            synthesisFiltersExpctd(:,:,:,1) = randn(2,2,2);
            synthesisFiltersExpctd(:,:,:,2) = randn(2,2,2);
            synthesisFiltersExpctd(:,:,:,3) = randn(2,2,2);
            synthesisFiltersExpctd(:,:,:,4) = randn(2,2,2);
            synthesisFiltersExpctd(:,:,:,5) = randn(2,2,2);
            synthesisFiltersExpctd(:,:,:,6) = randn(2,2,2);
            synthesisFiltersExpctd(:,:,:,7) = randn(2,2,2);
            synthesisFiltersExpctd(:,:,:,8) = randn(2,2,2);            
            
            % Instantiation
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'SynthesisFilters',synthesisFiltersExpctd);
            
            % Actual value
            synthesisFiltersActual = get(testCase.synthesizer,'SynthesisFilters');
            
            % Evaluation
            nChs = size(synthesisFiltersExpctd,3);
            for iCh = 1:nChs
                testCase.assertEqual(synthesisFiltersActual(:,:,:,iCh),...
                    synthesisFiltersExpctd(:,:,:,iCh));
            end
            
        end
        
        % Test
        function testStepDec222Ch44Ord000Level1(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            nDecs = [ 2 2 2 ];
            synthesisFilters(:,:,:,1) = randn(2,2,2);
            synthesisFilters(:,:,:,2) = randn(2,2,2);
            synthesisFilters(:,:,:,3) = randn(2,2,2);
            synthesisFilters(:,:,:,4) = randn(2,2,2);
            synthesisFilters(:,:,:,5) = randn(2,2,2);
            synthesisFilters(:,:,:,6) = randn(2,2,2);
            synthesisFilters(:,:,:,7) = randn(2,2,2);
            synthesisFilters(:,:,:,8) = randn(2,2,2);            
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            phase = [1 1 1]; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,:,iCh);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iCh},nDecs,phase),...
                    f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
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
        function testStepDec222Ch54Ord000Level1(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            nDecs = [ 2 2 2 ];
            synthesisFilters(:,:,:,1) = randn(2,2,2);
            synthesisFilters(:,:,:,2) = randn(2,2,2);
            synthesisFilters(:,:,:,3) = randn(2,2,2);
            synthesisFilters(:,:,:,4) = randn(2,2,2);
            synthesisFilters(:,:,:,5) = randn(2,2,2);
            synthesisFilters(:,:,:,6) = randn(2,2,2);
            synthesisFilters(:,:,:,7) = randn(2,2,2);
            synthesisFilters(:,:,:,8) = randn(2,2,2);            
            synthesisFilters(:,:,:,9) = randn(2,2,2);            
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            phase = [1 1 1]; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,:,iCh);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iCh},nDecs,phase),...
                    f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
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
            nDecs = [ 2 2 2 ];
            synthesisFilters(:,:,:,1) = randn(6,6,6);
            synthesisFilters(:,:,:,2) = randn(6,6,6);
            synthesisFilters(:,:,:,3) = randn(6,6,6);
            synthesisFilters(:,:,:,4) = randn(6,6,6);
            synthesisFilters(:,:,:,5) = randn(6,6,6);
            synthesisFilters(:,:,:,6) = randn(6,6,6);
            synthesisFilters(:,:,:,7) = randn(6,6,6);
            synthesisFilters(:,:,:,8) = randn(6,6,6);
            synthesisFilters(:,:,:,9) = randn(6,6,6);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            phase = [1 1 1]; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,:,iCh);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iCh},nDecs,phase),...
                    f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
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
            synthesisFilters(:,:,:,1) = randn(2,2,2);
            synthesisFilters(:,:,:,2) = randn(2,2,2);
            synthesisFilters(:,:,:,3) = randn(2,2,2);
            synthesisFilters(:,:,:,4) = randn(2,2,2);
            synthesisFilters(:,:,:,5) = randn(2,2,2);
            synthesisFilters(:,:,:,6) = randn(2,2,2);
            synthesisFilters(:,:,:,7) = randn(2,2,2);
            synthesisFilters(:,:,:,8) = randn(2,2,2);
            synthesisFilters(:,:,:,9) = randn(2,2,2);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            for iCh = 1:nChs
                f = synthesisFilters(:,:,:,iCh);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iCh},nDecs,phase),...
                    f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
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
            synthesisFilters(:,:,:,1) = randn(9,6,3);
            synthesisFilters(:,:,:,2) = randn(9,6,3);
            synthesisFilters(:,:,:,3) = randn(9,6,3);
            synthesisFilters(:,:,:,4) = randn(9,6,3);
            synthesisFilters(:,:,:,5) = randn(9,6,3);
            synthesisFilters(:,:,:,6) = randn(9,6,3);
            synthesisFilters(:,:,:,7) = randn(9,6,3);
            synthesisFilters(:,:,:,8) = randn(9,6,3);
            synthesisFilters(:,:,:,9) = randn(9,6,3);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            for iCh = 1:nChs
                f = synthesisFilters(:,:,:,iCh);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iCh},nDecs,phase),...
                    f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
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
            nDecs  = [ 2 2 2 ];
            synthesisFilters(:,:,:,1) = randn(6,6,6);
            synthesisFilters(:,:,:,2) = randn(6,6,6);
            synthesisFilters(:,:,:,3) = randn(6,6,6);
            synthesisFilters(:,:,:,4) = randn(6,6,6);
            synthesisFilters(:,:,:,5) = randn(6,6,6);
            synthesisFilters(:,:,:,6) = randn(6,6,6);
            synthesisFilters(:,:,:,7) = randn(6,6,6);
            synthesisFilters(:,:,:,8) = randn(6,6,6);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            phase = [ 1 1 1 ]; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,:,1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs{1},nDecs,phase),...
                    f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},nDecs,phase),...
                        f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
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
        function testStepDec222Ch44Ord222Level3(testCase)
            
            % Parameters
            height = 48;
            width  = 64;
            depth  = 32;
            nDecs  = [ 2 2 2 ];
            synthesisFilters(:,:,:,1) = randn(6,6,6);
            synthesisFilters(:,:,:,2) = randn(6,6,6);
            synthesisFilters(:,:,:,3) = randn(6,6,6);
            synthesisFilters(:,:,:,4) = randn(6,6,6);
            synthesisFilters(:,:,:,5) = randn(6,6,6);
            synthesisFilters(:,:,:,6) = randn(6,6,6);
            synthesisFilters(:,:,:,7) = randn(6,6,6);
            synthesisFilters(:,:,:,8) = randn(6,6,6);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            phase = [ 1 1 1 ]; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,:,1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs{1},nDecs,phase),...
                    f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},nDecs,phase),...
                        f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
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
        function testStepDec222Ch44Ord000Level1Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            nDecs = [ 2 2 2 ];
            synthesisFilters(:,:,:,1) = randn(2,2,2);
            synthesisFilters(:,:,:,2) = randn(2,2,2);
            synthesisFilters(:,:,:,3) = randn(2,2,2);
            synthesisFilters(:,:,:,4) = randn(2,2,2);
            synthesisFilters(:,:,:,5) = randn(2,2,2);
            synthesisFilters(:,:,:,6) = randn(2,2,2);
            synthesisFilters(:,:,:,7) = randn(2,2,2);
            synthesisFilters(:,:,:,8) = randn(2,2,2);            
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            phase = [1 1 1]; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,:,iCh);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iCh},nDecs,phase),...
                    f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            
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
        function testStepDec222Ch54Ord000Level1Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            nDecs = [ 2 2 2 ];
            synthesisFilters(:,:,:,1) = randn(2,2,2);
            synthesisFilters(:,:,:,2) = randn(2,2,2);
            synthesisFilters(:,:,:,3) = randn(2,2,2);
            synthesisFilters(:,:,:,4) = randn(2,2,2);
            synthesisFilters(:,:,:,5) = randn(2,2,2);
            synthesisFilters(:,:,:,6) = randn(2,2,2);
            synthesisFilters(:,:,:,7) = randn(2,2,2);
            synthesisFilters(:,:,:,8) = randn(2,2,2);            
            synthesisFilters(:,:,:,9) = randn(2,2,2);            
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            phase = [1 1 1]; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,:,iCh);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iCh},nDecs,phase),...
                    f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            
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
        function testStepDec222Ch54Ord222Level1Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            nDecs = [ 2 2 2 ];
            synthesisFilters(:,:,:,1) = randn(6,6,6);
            synthesisFilters(:,:,:,2) = randn(6,6,6);
            synthesisFilters(:,:,:,3) = randn(6,6,6);
            synthesisFilters(:,:,:,4) = randn(6,6,6);
            synthesisFilters(:,:,:,5) = randn(6,6,6);
            synthesisFilters(:,:,:,6) = randn(6,6,6);
            synthesisFilters(:,:,:,7) = randn(6,6,6);
            synthesisFilters(:,:,:,8) = randn(6,6,6);
            synthesisFilters(:,:,:,9) = randn(6,6,6);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            phase = [1 1 1]; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,:,iCh);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iCh},nDecs,phase),...
                    f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            
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
        function testStepDec111Ch54Ord111Level1Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            nDecs = [ 1 1 1 ];
            synthesisFilters(:,:,:,1) = randn(2,2,2);
            synthesisFilters(:,:,:,2) = randn(2,2,2);
            synthesisFilters(:,:,:,3) = randn(2,2,2);
            synthesisFilters(:,:,:,4) = randn(2,2,2);
            synthesisFilters(:,:,:,5) = randn(2,2,2);
            synthesisFilters(:,:,:,6) = randn(2,2,2);
            synthesisFilters(:,:,:,7) = randn(2,2,2);
            synthesisFilters(:,:,:,8) = randn(2,2,2);
            synthesisFilters(:,:,:,9) = randn(2,2,2);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            for iCh = 1:nChs
                f = synthesisFilters(:,:,:,iCh);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iCh},nDecs,phase),...
                    f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            
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
        function testStepDec321Ch44Ord222Level1Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            depth = 32;
            nDecs = [ 3 2 1 ];
            synthesisFilters(:,:,:,1) = randn(9,6,3);
            synthesisFilters(:,:,:,2) = randn(9,6,3);
            synthesisFilters(:,:,:,3) = randn(9,6,3);
            synthesisFilters(:,:,:,4) = randn(9,6,3);
            synthesisFilters(:,:,:,5) = randn(9,6,3);
            synthesisFilters(:,:,:,6) = randn(9,6,3);
            synthesisFilters(:,:,:,7) = randn(9,6,3);
            synthesisFilters(:,:,:,8) = randn(9,6,3);
            synthesisFilters(:,:,:,9) = randn(9,6,3);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            for iCh = 1:nChs
                f = synthesisFilters(:,:,:,iCh);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iCh},nDecs,phase),...
                    f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            
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
        function testStepDec222Ch44Ord222Level2Freq(testCase)
            
            % Parameters
            height = 48;
            width  = 64;
            depth  = 32;
            nDecs  = [ 2 2 2 ];
            synthesisFilters(:,:,:,1) = randn(6,6,6);
            synthesisFilters(:,:,:,2) = randn(6,6,6);
            synthesisFilters(:,:,:,3) = randn(6,6,6);
            synthesisFilters(:,:,:,4) = randn(6,6,6);
            synthesisFilters(:,:,:,5) = randn(6,6,6);
            synthesisFilters(:,:,:,6) = randn(6,6,6);
            synthesisFilters(:,:,:,7) = randn(6,6,6);
            synthesisFilters(:,:,:,8) = randn(6,6,6);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            phase = [ 1 1 1 ]; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,:,1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs{1},nDecs,phase),...
                    f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},nDecs,phase),...
                        f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord222Level3Freq(testCase)
            
            % Parameters
            height = 48;
            width  = 64;
            depth  = 32;
            nDecs  = [ 2 2 2 ];
            synthesisFilters(:,:,:,1) = randn(6,6,6);
            synthesisFilters(:,:,:,2) = randn(6,6,6);
            synthesisFilters(:,:,:,3) = randn(6,6,6);
            synthesisFilters(:,:,:,4) = randn(6,6,6);
            synthesisFilters(:,:,:,5) = randn(6,6,6);
            synthesisFilters(:,:,:,6) = randn(6,6,6);
            synthesisFilters(:,:,:,7) = randn(6,6,6);
            synthesisFilters(:,:,:,8) = randn(6,6,6);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            phase = [ 1 1 1 ]; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,:,1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs{1},nDecs,phase),...
                    f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},nDecs,phase),...
                        f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end

        function testStepDec234Ch1414Ord222Level1(testCase)
            
            % Parameters
            height = 8*2;
            width  = 12*3;
            depth  = 16*4;
            nDecs  = [ 2 3 4 ];
            synthesisFilters = zeros(6,9,12,28);
            for iCh = 1:28
                synthesisFilters(:,:,:,iCh) = randn(6,9,12);
            end
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            phase = [ 1 0 1 ]; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,:,1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs{1},nDecs,phase),...
                    f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},nDecs,phase),...
                        f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
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
            synthesisFilters = zeros(6,9,12,28);
            for iCh = 1:28
                synthesisFilters(:,:,:,iCh) = randn(6,9,12);
            end
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
                f = synthesisFilters(:,:,:,1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs{1},nDecs,phase),...
                    f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},nDecs,phase),...
                        f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
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

        %Test
        function testStepDec234Ch1414Ord222Level2Freq(testCase)
            
            % Parameters
            height = 8*2^2;
            width  = 12*3^2;
            depth  = 16*4^2;
            nDecs  = [ 2 3 4 ];
            synthesisFilters = zeros(6,9,12,28);
            for iCh = 1:28
                synthesisFilters(:,:,:,iCh) = randn(6,9,12);
            end
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
                f = synthesisFilters(:,:,:,1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs{1},nDecs,phase),...
                    f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},nDecs,phase),...
                        f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
    %Test
        function testStepDec234Ch1414Ord222Level2FreqUseGpuFalse(testCase)
            
            % Parameters
            height = 8*2^2;
            width  = 12*3^2;
            depth  = 16*4^2;
            nDecs  = [ 2 3 4 ];
            useGpu = false;
            synthesisFilters = zeros(6,9,12,28);
            for iCh = 1:28
                synthesisFilters(:,:,:,iCh) = randn(6,9,12);
            end
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
                f = synthesisFilters(:,:,:,1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs{1},nDecs,phase),...
                    f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},nDecs,phase),...
                        f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency',...
                'UseGpu',useGpu);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testClone(testCase)
            
            % Parameters
            height = 48;
            width  = 64;
            depth  = 32;
            nDecs  = [ 2 2 2 ];
            synthesisFilters(:,:,:,1) = randn(6,6,6);
            synthesisFilters(:,:,:,2) = randn(6,6,6);
            synthesisFilters(:,:,:,3) = randn(6,6,6);
            synthesisFilters(:,:,:,4) = randn(6,6,6);
            synthesisFilters(:,:,:,5) = randn(6,6,6);
            synthesisFilters(:,:,:,6) = randn(6,6,6);
            synthesisFilters(:,:,:,7) = randn(6,6,6);
            synthesisFilters(:,:,:,8) = randn(6,6,6);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            decZ = nDecs(Direction.DEPTH);
            nChs = size(synthesisFilters,4);
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
            phase = [ 1 1 1 ]; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,:,1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs{1},nDecs,phase),...
                    f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},nDecs,phase),...
                        f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis3dSystem(...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain','Frequency');
            cloneSynthesizer = clone(testCase.synthesizer);
            
            % Actual values
            imgActual = step(cloneSynthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end        
        

        % Test
        function testStepDec222Level1(testCase,nsubrows,nsubcols,nsublays)
            
            filterdom = 'Spatial'; % TODO: 'Frequency'
            pordXY = 2;
            pordZ = 0;
            redundancy = 2;
            ndecsX = 2;
            ndecsY = 2;
            ndecsZ = 2;
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nDecs = [ ndecsY ndecsX ndecsZ ];
            %height = nsubrows * ndecsY;
            %width = nsubcols * ndecsX;
            %depth = nsublays * ndecsZ;

            % Filters in XY
            nChsXY = redundancy*ndecsY*ndecsX;
            nChsZ = ndecsZ;
            nChs = nChsXY * nChsZ;
            lenY = (pordXY+1)*ndecsY;
            lenX = (pordXY+1)*ndecsX;
            lenZ = (pordZ+1)*ndecsZ;                            
            synthesisFilters = zeros(lenY,lenX,lenZ,nChs);
            for iCh = 1:nChs
                synthesisFilters(:,:,:,iCh) = randn(lenY,lenX,lenZ);
            end
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.generalfb.*            
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
            for iCh = 1:nChs
                subImg = subCoefs{iCh};
                % Interpolation filter
                f = synthesisFilters(:,:,:,iCh);
                % Upsample in Z
                if size(subImg,3) == 1
                    u = cat(3,subImg,...
                        zeros(size(subImg,1),size(subImg,2),nDecs(3)-1));
                    v = circshift(u,[0 0 phase(3)]);                    
                else
                    v = ipermute(upsample(permute(subImg,...
                        [3,1,2]),nDecs(3),phase(3)),[3,1,2]);
                end
                % Upsample in X
                if size(v,2) == 1
                    u = cat(2,v,...
                        zeros(size(v,1),nDecs(2)-1,size(v,3)));
                    v = circshift(u,[0 phase(2) 0]);                                        
                else
                    v = ipermute(upsample(permute(v,...
                        [2,1,3]),nDecs(2),phase(2)),[2,1,3]);
                end
                % Upsample in Y
                if size(v,1) == 1
                    u = cat(1,v,...
                        zeros(nDecs(1)-1,size(v,2),size(v,3)));
                    tmpImg = circshift(u,[phase(1) 0 0]);                                                            
                else
                    tmpImg = upsample(v,nDecs(1),phase(1));
                end
                % Synthesize
                imgExpctd = imgExpctd + imfilter(tmpImg,f,'circ','conv');
            end

            % Instantiation of target class
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain',filterdom);
            
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
        function testStepDec112Level1(testCase,nsubrows,nsubcols,nsublays)
            
            filterdom = 'Spatial'; % TODO: 'Frequency'
            pordXY = 2;
            pordZ = 0;
            redundancy = 1;
            ndecsX = 1;
            ndecsY = 1;
            ndecsZ = 2;
            
            % Parameters
            import saivdr.dictionary.utility.Direction
            nDecs = [ ndecsY ndecsX ndecsZ ];
            %height = nsubrows * ndecsY;
            %width = nsubcols * ndecsX;
            %depth = nsublays * ndecsZ;

            % Filters in XY
            nChsXY = redundancy*ndecsY*ndecsX;
            nChsZ = ndecsZ;
            nChs = nChsXY * nChsZ;
            lenY = (pordXY+1)*ndecsY;
            lenX = (pordXY+1)*ndecsX;
            lenZ = (pordZ+1)*ndecsZ;                            
            synthesisFilters = zeros(lenY,lenX,lenZ,nChs);
            for iCh = 1:nChs
                synthesisFilters(:,:,:,iCh) = randn(lenY,lenX,lenZ);
            end
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.generalfb.*            
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
            for iCh = 1:nChs
                subImg = subCoefs{iCh};
                % Interpolation filter
                f = synthesisFilters(:,:,:,iCh);
                % Upsample in Z
                if size(subImg,3) == 1
                    u = cat(3,subImg,...
                        zeros(size(subImg,1),size(subImg,2),nDecs(3)-1));
                    v = circshift(u,[0 0 phase(3)]);                    
                else
                    v = ipermute(upsample(permute(subImg,...
                        [3,1,2]),nDecs(3),phase(3)),[3,1,2]);
                end
                % Upsample in X
                if size(v,2) == 1
                    u = cat(2,v,...
                        zeros(size(v,1),nDecs(2)-1,size(v,3)));
                    v = circshift(u,[0 phase(2) 0]);                                        
                else
                    v = ipermute(upsample(permute(v,...
                        [2,1,3]),nDecs(2),phase(2)),[2,1,3]);
                end
                % Upsample in Y
                if size(v,1) == 1
                    u = cat(1,v,...
                        zeros(nDecs(1)-1,size(v,2),size(v,3)));
                    tmpImg = circshift(u,[phase(1) 0 0]);                                                            
                else
                    tmpImg = upsample(v,nDecs(1),phase(1));
                end
                % Synthesize
                imgExpctd = imgExpctd + imfilter(tmpImg,f,'circ','conv');
            end

            % Instantiation of target class
            testCase.synthesizer = Synthesis3dSystem(...
                'DecimationFactor',nDecs,...
                'SynthesisFilters',synthesisFilters,...
                'FilterDomain',filterdom);
            
            % Actual values
            imgActual = ...
                testCase.synthesizer.step(coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end

        
    end
    
end

