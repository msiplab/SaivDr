classdef Synthesis2dSystemTestCase < matlab.unittest.TestCase
    %SYNTHESIS2DSYSTEMTESTCASE Test case for Synthesis2dSystem
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
            decimationFactorExpctd = [ 2 2 ];
            frmbdExpctd  = [];
            filterDomainExpctd = 'Spatial';
            boundaryOperationExpctd = 'Circular';            
            
            % Instantiation
            testCase.synthesizer = Synthesis2dSystem();
            
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
            synthesisFiltersExpctd(:,:,1) = randn(2,2);
            synthesisFiltersExpctd(:,:,2) = randn(2,2);
            synthesisFiltersExpctd(:,:,3) = randn(2,2);
            synthesisFiltersExpctd(:,:,4) = randn(2,2);
            
            % Instantiation
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
                'SynthesisFilters',synthesisFiltersExpctd);
            
            % Actual value
            synthesisFiltersActual = get(testCase.synthesizer,'SynthesisFilters');
            
            % Evaluation
            nChs = size(synthesisFiltersExpctd,3);
            for iCh = 1:nChs
                testCase.assertEqual(synthesisFiltersActual(:,:,iCh),...
                    synthesisFiltersExpctd(:,:,iCh));
            end
            
        end
        
        % Test
        function testStepDec22Ch22Ord00Level1(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            nDecs = [ 2 2 ];
            synthesisFilters(:,:,1) = randn(2,2);
            synthesisFilters(:,:,2) = randn(2,2);
            synthesisFilters(:,:,3) = randn(2,2);
            synthesisFilters(:,:,4) = randn(2,2);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(prod(nDecs),2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iCh}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec22Ch33Ord00Level1(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            nDecs = [ 2 2 ];
            synthesisFilters(:,:,1) = randn(2,2);
            synthesisFilters(:,:,2) = randn(2,2);
            synthesisFilters(:,:,3) = randn(2,2);
            synthesisFilters(:,:,4) = randn(2,2);
            synthesisFilters(:,:,5) = randn(2,2);
            synthesisFilters(:,:,6) = randn(2,2);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(prod(nDecs),2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iCh}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec22Ch33Ord22Level1(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            nDecs = [ 2 2 ];
            synthesisFilters(:,:,1) = randn(6,6);
            synthesisFilters(:,:,2) = randn(6,6);
            synthesisFilters(:,:,3) = randn(6,6);
            synthesisFilters(:,:,4) = randn(6,6);
            synthesisFilters(:,:,5) = randn(6,6);
            synthesisFilters(:,:,6) = randn(6,6);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(prod(nDecs),2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iCh}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec11Ch33Ord11Level1(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            nDecs = [ 1 1 ];
            synthesisFilters(:,:,1) = randn(2,2);
            synthesisFilters(:,:,2) = randn(2,2);
            synthesisFilters(:,:,3) = randn(2,2);
            synthesisFilters(:,:,4) = randn(2,2);
            synthesisFilters(:,:,5) = randn(2,2);
            synthesisFilters(:,:,6) = randn(2,2);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(prod(nDecs),2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iCh}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec33Ch54Ord22Level1(testCase)
            
            % Parameters
            height = 12*3;
            width = 16*3;
            nDecs = [ 3 3 ];
            synthesisFilters(:,:,1) = randn(9,9);
            synthesisFilters(:,:,2) = randn(9,9);
            synthesisFilters(:,:,3) = randn(9,9);
            synthesisFilters(:,:,4) = randn(9,9);
            synthesisFilters(:,:,5) = randn(9,9);
            synthesisFilters(:,:,6) = randn(9,9);
            synthesisFilters(:,:,7) = randn(9,9);
            synthesisFilters(:,:,8) = randn(9,9);
            synthesisFilters(:,:,9) = randn(9,9);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(prod(nDecs),2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iCh}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec44Ch1212Ord22Level1(testCase)
            
            % Parameters
            height = 12*4;
            width = 16*4;
            nDecs = [ 4 4 ];
            synthesisFilters = zeros(12,12,24);
            for iCh = 1:24
                synthesisFilters(:,:,iCh) = randn(12,12);
            end
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(prod(nDecs),2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iCh}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec22Ch22Ord22Level2(testCase)
            
            % Parameters
            height = 48;
            width  = 64;
            nDecs  = [ 2 2 ];
            synthesisFilters(:,:,1) = randn(6,6);
            synthesisFilters(:,:,2) = randn(6,6);
            synthesisFilters(:,:,3) = randn(6,6);
            synthesisFilters(:,:,4) = randn(6,6);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY^2),width/(decX^2));
            subCoefs{2} = rand(height/(decY^2),width/(decX^2));
            subCoefs{3} = rand(height/(decY^2),width/(decX^2));
            subCoefs{4} = rand(height/(decY^2),width/(decX^2));
            subCoefs{5} = rand(height/(decY),width/(decX));
            subCoefs{6} = rand(height/(decY),width/(decX));
            subCoefs{7} = rand(height/(decY),width/(decX));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec22Ch22Ord22Level3(testCase)
            % Parameters
            height = 48;
            width  = 64;
            nDecs  = [ 2 2 ];
            synthesisFilters(:,:,1) = randn(6,6);
            synthesisFilters(:,:,2) = randn(6,6);
            synthesisFilters(:,:,3) = randn(6,6);
            synthesisFilters(:,:,4) = randn(6,6);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY^3),width/(decX^3));
            subCoefs{2} = rand(height/(decY^3),width/(decX^3));
            subCoefs{3} = rand(height/(decY^3),width/(decX^3));
            subCoefs{4} = rand(height/(decY^3),width/(decX^3));
            subCoefs{5} = rand(height/(decY^2),width/(decX^2));
            subCoefs{6} = rand(height/(decY^2),width/(decX^2));
            subCoefs{7} = rand(height/(decY^2),width/(decX^2));
            subCoefs{8} = rand(height/(decY),width/(decX));
            subCoefs{9} = rand(height/(decY),width/(decX));
            subCoefs{10} = rand(height/(decY),width/(decX));            
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec22Ch22Ord00Level1Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            nDecs = [ 2 2 ];
            synthesisFilters(:,:,1) = randn(2,2);
            synthesisFilters(:,:,2) = randn(2,2);
            synthesisFilters(:,:,3) = randn(2,2);
            synthesisFilters(:,:,4) = randn(2,2);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(prod(nDecs),2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iCh}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec22Ch33Ord00Level1Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            nDecs = [ 2 2 ];
            synthesisFilters(:,:,1) = randn(2,2);
            synthesisFilters(:,:,2) = randn(2,2);
            synthesisFilters(:,:,3) = randn(2,2);
            synthesisFilters(:,:,4) = randn(2,2);
            synthesisFilters(:,:,5) = randn(2,2);
            synthesisFilters(:,:,6) = randn(2,2);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(prod(nDecs),2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iCh}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec22Ch334Ord22Level1Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            nDecs = [ 2 2 ];
            synthesisFilters(:,:,1) = randn(6,6);
            synthesisFilters(:,:,2) = randn(6,6);
            synthesisFilters(:,:,3) = randn(6,6);
            synthesisFilters(:,:,4) = randn(6,6);
            synthesisFilters(:,:,5) = randn(6,6);
            synthesisFilters(:,:,6) = randn(6,6);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(prod(nDecs),2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iCh}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec11Ch33Ord11Level1Freq(testCase)
            
            % Parameters
            height = 48;
            width = 64;
            nDecs = [ 1 1 ];
            synthesisFilters(:,:,1) = randn(2,2);
            synthesisFilters(:,:,2) = randn(2,2);
            synthesisFilters(:,:,3) = randn(2,2);
            synthesisFilters(:,:,4) = randn(2,2);
            synthesisFilters(:,:,5) = randn(2,2);
            synthesisFilters(:,:,6) = randn(2,2);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(prod(nDecs),2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iCh}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec33Ch54Ord22Level1Freq(testCase)
            
            % Parameters
            height = 12*3;
            width = 16*3;
            nDecs = [ 3 3 ];
            synthesisFilters(:,:,1) = randn(9,9);
            synthesisFilters(:,:,2) = randn(9,9);
            synthesisFilters(:,:,3) = randn(9,9);
            synthesisFilters(:,:,4) = randn(9,9);
            synthesisFilters(:,:,5) = randn(9,9);
            synthesisFilters(:,:,6) = randn(9,9);
            synthesisFilters(:,:,7) = randn(9,9);
            synthesisFilters(:,:,8) = randn(9,9);
            synthesisFilters(:,:,9) = randn(9,9);
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(prod(nDecs),2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iCh}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec44Ch1212Ord22Level1Freq(testCase)
            
            % Parameters
            height = 12*4;
            width = 16*4;
            nDecs = [ 4 4 ];
            synthesisFilters = zeros(12,12,24);
            for iCh = 1:24
                synthesisFilters(:,:,iCh) = randn(12,12);
            end
            %nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(prod(nDecs),2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = rand(height/decY,width/decX);
                subCoefs{iCh} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iCh = 1:nChs
                f = synthesisFilters(:,:,iCh);
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iCh}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec22Ch22Ord22Level2Freq(testCase)
            
            % Parameters
            height = 48;
            width  = 64;
            nDecs  = [ 2 2 ];
            synthesisFilters(:,:,1) = randn(6,6);
            synthesisFilters(:,:,2) = randn(6,6);
            synthesisFilters(:,:,3) = randn(6,6);
            synthesisFilters(:,:,4) = randn(6,6);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY^2),width/(decX^2));
            subCoefs{2} = rand(height/(decY^2),width/(decX^2));
            subCoefs{3} = rand(height/(decY^2),width/(decX^2));
            subCoefs{4} = rand(height/(decY^2),width/(decX^2));
            subCoefs{5} = rand(height/(decY),width/(decX));
            subCoefs{6} = rand(height/(decY),width/(decX));
            subCoefs{7} = rand(height/(decY),width/(decX));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec22Ch22Ord22Level3Freq(testCase)
            % Parameters
            height = 48;
            width  = 64;
            nDecs  = [ 2 2 ];
            synthesisFilters(:,:,1) = randn(6,6);
            synthesisFilters(:,:,2) = randn(6,6);
            synthesisFilters(:,:,3) = randn(6,6);
            synthesisFilters(:,:,4) = randn(6,6);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY^3),width/(decX^3));
            subCoefs{2} = rand(height/(decY^3),width/(decX^3));
            subCoefs{3} = rand(height/(decY^3),width/(decX^3));
            subCoefs{4} = rand(height/(decY^3),width/(decX^3));
            subCoefs{5} = rand(height/(decY^2),width/(decX^2));
            subCoefs{6} = rand(height/(decY^2),width/(decX^2));
            subCoefs{7} = rand(height/(decY^2),width/(decX^2));
            subCoefs{8} = rand(height/(decY),width/(decX));
            subCoefs{9} = rand(height/(decY),width/(decX));
            subCoefs{10} = rand(height/(decY),width/(decX));            
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        function testStepDec33Ch54Ord22Level2(testCase)
            % Parameters
            height = 108;
            width  = 135;
            nDecs  = [ 3 3 ];
            synthesisFilters(:,:,1) = randn(9,9);
            synthesisFilters(:,:,2) = randn(9,9);
            synthesisFilters(:,:,3) = randn(9,9);
            synthesisFilters(:,:,4) = randn(9,9);
            synthesisFilters(:,:,5) = randn(9,9);
            synthesisFilters(:,:,6) = randn(9,9);
            synthesisFilters(:,:,7) = randn(9,9);
            synthesisFilters(:,:,8) = randn(9,9);
            synthesisFilters(:,:,9) = randn(9,9);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY^2),width/(decX^2));
            subCoefs{2} = rand(height/(decY^2),width/(decX^2));
            subCoefs{3} = rand(height/(decY^2),width/(decX^2));
            subCoefs{4} = rand(height/(decY^2),width/(decX^2));
            subCoefs{5} = rand(height/(decY^2),width/(decX^2));
            subCoefs{6} = rand(height/(decY^2),width/(decX^2));
            subCoefs{7} = rand(height/(decY^2),width/(decX^2));
            subCoefs{8} = rand(height/(decY^2),width/(decX^2));
            subCoefs{9} = rand(height/(decY^2),width/(decX^2));
            subCoefs{10} = rand(height/(decY),width/(decX));
            subCoefs{11} = rand(height/(decY),width/(decX));
            subCoefs{12} = rand(height/(decY),width/(decX));
            subCoefs{13} = rand(height/(decY),width/(decX));
            subCoefs{14} = rand(height/(decY),width/(decX));            
            subCoefs{15} = rand(height/(decY),width/(decX));
            subCoefs{16} = rand(height/(decY),width/(decX));
            subCoefs{17} = rand(height/(decY),width/(decX));            
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 0; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
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
        function testStepDec44Ch1212Ord22Level2(testCase)
            % Parameters
            height = 12*4*4;
            width  = 16*4*4;
            nDecs  = [ 4 4 ];
            synthesisFilters = zeros(12,12,24);
            for iCh = 1:24
                synthesisFilters(:,:,iCh) = randn(12,12);
            end
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = randn(height/(decY^2),width/(decX^2));
            for iCh = 2:24
                subCoefs{iCh} = randn(height/(decY^2),width/(decX^2));
                subCoefs{iCh+23} = rand(height/(decY),width/(decX));
            end
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
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
        function testStepDec44Ch1212Ord22Level2Freq(testCase)
            % Parameters
            height = 12*4*4;
            width  = 16*4*4;
            nDecs  = [ 4 4 ];
            synthesisFilters = zeros(12,12,24);
            for iCh = 1:24
                synthesisFilters(:,:,iCh) = randn(12,12);
            end
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = randn(height/(decY^2),width/(decX^2));
            for iCh = 2:24
                subCoefs{iCh} = randn(height/(decY^2),width/(decX^2));
                subCoefs{iCh+23} = rand(height/(decY),width/(decX));
            end
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        
        % Test
        function testStepDec33Ch54Ord22Level3(testCase)
            % Parameters
            height = 108;
            width  = 135;
            nDecs  = [ 3 3 ];
            synthesisFilters(:,:,1) = randn(6,6);
            synthesisFilters(:,:,2) = randn(6,6);
            synthesisFilters(:,:,3) = randn(6,6);
            synthesisFilters(:,:,4) = randn(6,6);
            synthesisFilters(:,:,5) = randn(6,6);
            synthesisFilters(:,:,6) = randn(6,6);
            synthesisFilters(:,:,7) = randn(6,6);
            synthesisFilters(:,:,8) = randn(6,6);            
            synthesisFilters(:,:,9) = randn(6,6);            
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY^3),width/(decX^3));
            subCoefs{2} = rand(height/(decY^3),width/(decX^3));
            subCoefs{3} = rand(height/(decY^3),width/(decX^3));
            subCoefs{4} = rand(height/(decY^3),width/(decX^3));
            subCoefs{5} = rand(height/(decY^3),width/(decX^3));
            subCoefs{6} = rand(height/(decY^3),width/(decX^3));
            subCoefs{7} = rand(height/(decY^3),width/(decX^3));
            subCoefs{8} = rand(height/(decY^3),width/(decX^3));
            subCoefs{9} = rand(height/(decY^3),width/(decX^3));
            subCoefs{10} = rand(height/(decY^2),width/(decX^2));
            subCoefs{11} = rand(height/(decY^2),width/(decX^2));
            subCoefs{12} = rand(height/(decY^2),width/(decX^2));
            subCoefs{13} = rand(height/(decY^2),width/(decX^2));
            subCoefs{14} = rand(height/(decY^2),width/(decX^2));            
            subCoefs{15} = rand(height/(decY^2),width/(decX^2));
            subCoefs{16} = rand(height/(decY^2),width/(decX^2));
            subCoefs{17} = rand(height/(decY^2),width/(decX^2));            
            subCoefs{18} = rand(height/(decY),width/(decX));
            subCoefs{19} = rand(height/(decY),width/(decX));
            subCoefs{20} = rand(height/(decY),width/(decX));            
            subCoefs{21} = rand(height/(decY),width/(decX));
            subCoefs{22} = rand(height/(decY),width/(decX));
            subCoefs{23} = rand(height/(decY),width/(decX));            
            subCoefs{24} = rand(height/(decY),width/(decX));
            subCoefs{25} = rand(height/(decY),width/(decX));                        
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 0; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
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
        function testStepDec33Ch54Ord22Level2Freq(testCase)
            % Parameters
            height = 108;
            width  = 135;
            nDecs  = [ 3 3 ];
            synthesisFilters(:,:,1) = randn(6,6);
            synthesisFilters(:,:,2) = randn(6,6);
            synthesisFilters(:,:,3) = randn(6,6);
            synthesisFilters(:,:,4) = randn(6,6);
            synthesisFilters(:,:,5) = randn(6,6);
            synthesisFilters(:,:,6) = randn(6,6);
            synthesisFilters(:,:,7) = randn(6,6);
            synthesisFilters(:,:,8) = randn(6,6);            
            synthesisFilters(:,:,9) = randn(6,6);            
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY^2),width/(decX^2));
            subCoefs{2} = rand(height/(decY^2),width/(decX^2));
            subCoefs{3} = rand(height/(decY^2),width/(decX^2));
            subCoefs{4} = rand(height/(decY^2),width/(decX^2));
            subCoefs{5} = rand(height/(decY^2),width/(decX^2));
            subCoefs{6} = rand(height/(decY^2),width/(decX^2));
            subCoefs{7} = rand(height/(decY^2),width/(decX^2));
            subCoefs{8} = rand(height/(decY^2),width/(decX^2));
            subCoefs{9} = rand(height/(decY^2),width/(decX^2));
            subCoefs{10} = rand(height/(decY),width/(decX));
            subCoefs{11} = rand(height/(decY),width/(decX));
            subCoefs{12} = rand(height/(decY),width/(decX));
            subCoefs{13} = rand(height/(decY),width/(decX));
            subCoefs{14} = rand(height/(decY),width/(decX));            
            subCoefs{15} = rand(height/(decY),width/(decX));
            subCoefs{16} = rand(height/(decY),width/(decX));
            subCoefs{17} = rand(height/(decY),width/(decX));            
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 0; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        
        % Test
        function testStepDec33Ch54Ord22Level3Freq(testCase)
            % Parameters
            height = 108;
            width  = 135;
            nDecs  = [ 3 3 ];
            synthesisFilters(:,:,1) = randn(9,9);
            synthesisFilters(:,:,2) = randn(9,9);
            synthesisFilters(:,:,3) = randn(9,9);
            synthesisFilters(:,:,4) = randn(9,9);
            synthesisFilters(:,:,5) = randn(9,9);
            synthesisFilters(:,:,6) = randn(9,9);
            synthesisFilters(:,:,7) = randn(9,9);
            synthesisFilters(:,:,8) = randn(9,9);
            synthesisFilters(:,:,9) = randn(9,9);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY^3),width/(decX^3));
            subCoefs{2} = rand(height/(decY^3),width/(decX^3));
            subCoefs{3} = rand(height/(decY^3),width/(decX^3));
            subCoefs{4} = rand(height/(decY^3),width/(decX^3));
            subCoefs{5} = rand(height/(decY^3),width/(decX^3));
            subCoefs{6} = rand(height/(decY^3),width/(decX^3));
            subCoefs{7} = rand(height/(decY^3),width/(decX^3));
            subCoefs{8} = rand(height/(decY^3),width/(decX^3));
            subCoefs{9} = rand(height/(decY^3),width/(decX^3));
            subCoefs{10} = rand(height/(decY^2),width/(decX^2));
            subCoefs{11} = rand(height/(decY^2),width/(decX^2));
            subCoefs{12} = rand(height/(decY^2),width/(decX^2));
            subCoefs{13} = rand(height/(decY^2),width/(decX^2));
            subCoefs{14} = rand(height/(decY^2),width/(decX^2));            
            subCoefs{15} = rand(height/(decY^2),width/(decX^2));
            subCoefs{16} = rand(height/(decY^2),width/(decX^2));
            subCoefs{17} = rand(height/(decY^2),width/(decX^2));            
            subCoefs{18} = rand(height/(decY),width/(decX));
            subCoefs{19} = rand(height/(decY),width/(decX));
            subCoefs{20} = rand(height/(decY),width/(decX));            
            subCoefs{21} = rand(height/(decY),width/(decX));
            subCoefs{22} = rand(height/(decY),width/(decX));
            subCoefs{23} = rand(height/(decY),width/(decX));            
            subCoefs{24} = rand(height/(decY),width/(decX));
            subCoefs{25} = rand(height/(decY),width/(decX));                        
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 0; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        
        % Test
        function testStepDec44Ch1212Ord22Level3(testCase)
            % Parameters
            height = 12*4^3;
            width  = 16*4^3;
            nDecs  = [ 4 4 ];
            synthesisFilters = zeros(12,12,24);
            for iCh = 1:24
                synthesisFilters(:,:,iCh) = randn(12,12);
            end
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = randn(height/(decY^3),width/(decX^3));
            for iCh = 2:24
                subCoefs{iCh}    = randn(height/(decY^3),width/(decX^3));
                subCoefs{iCh+23} = randn(height/(decY^2),width/(decX^2));
                subCoefs{iCh+46} = randn(height/(decY),width/(decX));
            end
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
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
        function testStepDec44Ch1212Ord22Level3Freq(testCase)
            % Parameters
            height = 12*4^3;
            width  = 16*4^3;
            nDecs  = [ 4 4 ];
            synthesisFilters = zeros(12,12,24);
            for iCh = 1:24
                synthesisFilters(:,:,iCh) = randn(12,12);
            end
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = randn(height/(decY^3),width/(decX^3));
            for iCh = 2:24
                subCoefs{iCh}    = randn(height/(decY^3),width/(decX^3));
                subCoefs{iCh+23} = randn(height/(decY^2),width/(decX^2));
                subCoefs{iCh+46} = randn(height/(decY),width/(decX));
            end
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
        
        % Test
        function testStepDec44Ch1212Ord22Level3FreqUseGpuFalse(testCase)
            % Parameters
            height = 12*4^3;
            width  = 16*4^3;
            nDecs  = [ 4 4 ];
            useGpu = false;
            synthesisFilters = zeros(12,12,24);
            for iCh = 1:24
                synthesisFilters(:,:,iCh) = randn(12,12);
            end
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = randn(height/(decY^3),width/(decX^3));
            for iCh = 2:24
                subCoefs{iCh}    = randn(height/(decY^3),width/(decX^3));
                subCoefs{iCh+23} = randn(height/(decY^2),width/(decX^2));
                subCoefs{iCh+46} = randn(height/(decY),width/(decX));
            end
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
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
            height = 108;
            width  = 135;
            nDecs  = [ 3 3 ];
            synthesisFilters(:,:,1) = randn(6,6);
            synthesisFilters(:,:,2) = randn(6,6);
            synthesisFilters(:,:,3) = randn(6,6);
            synthesisFilters(:,:,4) = randn(6,6);
            synthesisFilters(:,:,5) = randn(6,6);
            synthesisFilters(:,:,6) = randn(6,6);
            synthesisFilters(:,:,7) = randn(6,6);
            synthesisFilters(:,:,8) = randn(6,6);            
            synthesisFilters(:,:,9) = randn(6,6);            
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.utility.Direction
            decY = nDecs(Direction.VERTICAL);
            decX = nDecs(Direction.HORIZONTAL);
            nChs = size(synthesisFilters,3);
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decY^3),width/(decX^3));
            subCoefs{2} = rand(height/(decY^3),width/(decX^3));
            subCoefs{3} = rand(height/(decY^3),width/(decX^3));
            subCoefs{4} = rand(height/(decY^3),width/(decX^3));
            subCoefs{5} = rand(height/(decY^3),width/(decX^3));
            subCoefs{6} = rand(height/(decY^3),width/(decX^3));
            subCoefs{7} = rand(height/(decY^3),width/(decX^3));
            subCoefs{8} = rand(height/(decY^3),width/(decX^3));
            subCoefs{9} = rand(height/(decY^3),width/(decX^3));
            subCoefs{10} = rand(height/(decY^2),width/(decX^2));
            subCoefs{11} = rand(height/(decY^2),width/(decX^2));
            subCoefs{12} = rand(height/(decY^2),width/(decX^2));
            subCoefs{13} = rand(height/(decY^2),width/(decX^2));
            subCoefs{14} = rand(height/(decY^2),width/(decX^2));            
            subCoefs{15} = rand(height/(decY^2),width/(decX^2));
            subCoefs{16} = rand(height/(decY^2),width/(decX^2));
            subCoefs{17} = rand(height/(decY^2),width/(decX^2));            
            subCoefs{18} = rand(height/(decY),width/(decX));
            subCoefs{19} = rand(height/(decY),width/(decX));
            subCoefs{20} = rand(height/(decY),width/(decX));            
            subCoefs{21} = rand(height/(decY),width/(decX));
            subCoefs{22} = rand(height/(decY),width/(decX));
            subCoefs{23} = rand(height/(decY),width/(decX));            
            subCoefs{24} = rand(height/(decY),width/(decX));
            subCoefs{25} = rand(height/(decY),width/(decX));                        
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Expected values
            phase = 0; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                f = synthesisFilters(:,:,1);
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decX,phase).',...
                    decY,phase),f,'conv','circ');
                for iCh = 2:nChs
                    f = synthesisFilters(:,:,iCh);
                    iSubband = (iLevel-1)*(nChs-1)+iCh;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decX,phase).',...
                        decY,phase),f,'conv','circ');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.generalfb.*
            testCase.synthesizer = Synthesis2dSystem(...
                'DecimationFactor',nDecs,...
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
    end
    
end
