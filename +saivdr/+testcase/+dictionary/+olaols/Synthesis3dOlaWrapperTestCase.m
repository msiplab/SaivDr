classdef Synthesis3dOlaWrapperTestCase < matlab.unittest.TestCase
    %SYNTHESIS3DOLASYSTEMTESTCASE Test case for Synthesis3dOlaWrapper
    %
    % Requirements: MATLAB R2018a
    %
    % Copyright (c) 2018, Shogo MURAMATSU
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
            import saivdr.dictionary.olaols.*
            synthesizerExpctd = [];
            boundaryOperationExpctd = [];
            
            % Instantiation
            testCase.synthesizer = Synthesis3dOlaWrapper();
            
            % Actual value
            synthesizerActual = testCase.synthesizer.Synthesizer;
            boundaryOperationActual = testCase.synthesizer.BoundaryOperation;
            
            % Evaluation
            testCase.assertEqual(synthesizerActual,synthesizerExpctd);
            testCase.assertEqual(boundaryOperationActual,boundaryOperationExpctd);
        end
        
        
        % Test
        function testSynthesizer(testCase)
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            synthesizerExpctd = UdHaarSynthesis3dSystem();
            
            % Instantiation
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',synthesizerExpctd);
            
            % Actual value
            synthesizerActual = get(testCase.synthesizer,'Synthesizer');
            
            % Evaluation
            testCase.assertEqual(synthesizerActual, synthesizerExpctd);
            
        end
        
        
        % Test
        function testUdHaarLevel1(testCase,width,height,depth)
            
            nLevels = 1;
            caa = rand(height,width,depth);
            cha  = rand(height,width,depth);
            cva  = rand(height,width,depth);
            cda  = rand(height,width,depth);
            cad  = rand(height,width,depth);
            chd  = rand(height,width,depth);
            cvd  = rand(height,width,depth);
            cdd  = rand(height,width,depth);
            subCoefs = [
                caa(:)
                cha(:)
                cva(:)
                cda(:)
                cad(:)
                chd(:)
                cvd(:)
                cdd(:) ].';
            scales= repmat([ height width depth],[7*nLevels+1, 1]);
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            imgExpctd = step(refSynthesizer,subCoefs,scales);
            dimExpctd = [ height width depth ];
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer);
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,...
                'RelTol',1e-7,sprintf('%g',diff));
        end
        
        
        % Test
        function testUdHaarLevel2(testCase,width,height,depth)
            
            % Parameters
            nLevels = 2;
            caa1 = rand(height,width,depth);
            cha1 = rand(height,width,depth);
            cva1 = rand(height,width,depth);
            cda1 = rand(height,width,depth);
            cad1 = rand(height,width,depth);
            chd1 = rand(height,width,depth);
            cvd1 = rand(height,width,depth);
            cdd1 = rand(height,width,depth);
            cha2 = rand(height,width,depth);
            cva2 = rand(height,width,depth);
            cda2 = rand(height,width,depth);
            cad2 = rand(height,width,depth);
            chd2 = rand(height,width,depth);
            cvd2 = rand(height,width,depth);
            cdd2 = rand(height,width,depth);
            subCoefs = [
                caa1(:)
                cha1(:)
                cva1(:)
                cda1(:)
                cad1(:)
                chd1(:)
                cvd1(:)
                cdd1(:)
                cha2(:)
                cva2(:)
                cda2(:)
                cad2(:)
                chd2(:)
                cvd2(:)
                cdd2(:) ].';
            scales = repmat([ height width depth],[7*nLevels+1, 1]);
            
            
            % Expected values
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            imgExpctd = step(refSynthesizer,subCoefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer);
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testUdHaar(testCase,width,height,depth,level)
            % Parameters
            nChs = 7*level+1;
            subCoefs = repmat(rand(1,height*width*depth),[1 nChs]);
            scales = repmat([ height width depth ],[7*level+1, 1]);
            
            % Preparation
            % Expected values
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            imgExpctd = step(refSynthesizer,subCoefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer);
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testUdHaarCellInput(testCase,width,height,depth,level)
            % Parameters
            nChs = 7*level+1;
            nSplit = 1;
            subCoefs = cell(nSplit,1);
            subCoefs{1} = repmat(rand(1,height*width*depth),[1 nChs]);
            scales = repmat([ height width depth ],[7*level+1, 1]);
            
            % Preparation
            % Expected values
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            imgExpctd = step(refSynthesizer,subCoefs{1},scales);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer,...
                'InputType','Cell');
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        
        % Test
        function testUdHaarSplittingSize(testCase,width,height,depth,useparallel)
            
            % Parameters
            nLevels = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(nLevels-1)-1;
            nHorPad = 2^(nLevels-1)-1;
            nDepPad = 2^(nLevels-1)-1;
            nChs = 7*nLevels+1;
            subCoefs = repmat(rand(1,height*width*depth),[1 nChs]);
            scales = repmat([ height width depth],[7*nLevels+1, 1]);
            
            % Preparation
            % Expected values
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            imgExpctd = step(refSynthesizer,subCoefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scales);
            
            % Evaluation
            %testCase.assertFail('TODO: Check for split');
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testUdHaarSplittingSplit(testCase,vsplit,hsplit,dsplit,useparallel)
            
            % Parameters
            nLevels = 2;
            height_ = 48;
            width_ = 48;
            depth_ = 48;
            nVerSplit = vsplit;
            nHorSplit = hsplit;
            nDepSplit = dsplit;
            nVerPad = 2^(nLevels-1)-1;
            nHorPad = 2^(nLevels-1)-1;
            nDepPad = 2^(nLevels-1)-1;
            nChs = 7*nLevels+1;
            subCoefs = repmat(rand(1,height_*width_*depth_),[1 nChs]);
            scales = repmat([ height_ width_ depth_],[7*nLevels+1, 1]);
            
            % Preparation
            % Expected values
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            imgExpctd = step(refSynthesizer,subCoefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scales);
            
            % Evaluation
            %testCase.assertFail('TODO: Check for split');
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
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
            nVerPad = 2^(level-1)-1;
            nHorPad = 2^(level-1)-1;
            nDepPad = 2^(level-1)-1;
            nChs = 7*level+1;
            subCoefs = repmat(rand(1,height_*width_*depth_),[1 nChs]);
            scales = repmat([ height_ width_ depth_],[7*level+1, 1]);
            
            % Preparation
            % Expected values
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            imgExpctd = step(refSynthesizer,subCoefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scales);
            
            % Evaluation
            %testCase.assertFail('TODO: Check for split');
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        
        % Test
        function testUdHaarSplittingCellInputSize(testCase,width,height,depth,...
                useparallel)
            
            % Parameters
            nLevels = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            nDepPad = 2^(nLevels-1);
            nChs = 7*nLevels+1;
            nSplit = nVerSplit*nHorSplit*nDepSplit;
            subCoefArrays = cell(nSplit,nChs);
            subCoefs = cell(nSplit,1);
            for iSplit = 1:nSplit
                subCoefs{iSplit} = [];
                for iCh = 1:nChs
                    coefVec = rand(1,height*width*depth/nSplit);
                    subCoefArrays{iSplit,iCh} = reshape(coefVec,...
                        [height/nVerSplit width/nHorSplit depth/nDepSplit]);
                    subCoefs{iSplit} = [ subCoefs{iSplit} coefVec ];
                end
            end
            subScales = repmat([ height/nVerSplit width/nHorSplit depth/nDepSplit],...
                [7*nLevels+1, 1]);
            
            % Preparation
            % Expected values
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            refCoefs = [];
            tmpArrays = cell(nVerSplit,nHorSplit,nDepSplit);
            for iCh = 1:nChs
                iSplit = 0;
                for iLay = 1:nDepSplit
                    for iCol = 1:nHorSplit
                        for iRow = 1:nVerSplit
                            iSplit = iSplit + 1;
                            tmpArrays{iRow,iCol,iLay} = subCoefArrays{iSplit,iCh};
                        end
                    end
                end
                tmpArray = cell2mat(tmpArrays);
                refCoefs = [ refCoefs tmpArray(:).'];
            end
            refScales = repmat([ height width depth ],[7*nLevels+1, 1]);
            imgExpctd = step(refSynthesizer,refCoefs,refScales);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel,...
                'InputType','Cell');
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,subScales);
            
            % Evaluation
            %testCase.assertFail('TODO: Check for split');
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testUdHaarSplittingCellInputSplit(testCase,...
                vsplit,hsplit,dsplit,useparallel)
            
            % Parameters
            height_ = 48;
            width_ = 48;
            depth_ = 48;
            nLevels =  2;
            nVerSplit = vsplit;
            nHorSplit = hsplit;
            nDepSplit = dsplit;
            nVerPad = 2^(nLevels-1);
            nHorPad = 2^(nLevels-1);
            nDepPad = 2^(nLevels-1);
            nChs = 7*nLevels+1;
            nSplit = nVerSplit*nHorSplit*nDepSplit;
            subCoefArrays = cell(nSplit,nChs);
            subCoefs = cell(nSplit,1);
            for iSplit = 1:nSplit
                subCoefs{iSplit} = [];
                for iCh = 1:nChs
                    coefVec = rand(1,height_*width_*depth_/nSplit);
                    subCoefArrays{iSplit,iCh} = reshape(coefVec,...
                        [height_/nVerSplit width_/nHorSplit depth_/nDepSplit]);
                    subCoefs{iSplit} = [ subCoefs{iSplit} coefVec ];
                end
            end
            subScales = repmat([ height_/nVerSplit width_/nHorSplit depth_/nDepSplit],...
                [7*nLevels+1, 1]);
            
            % Preparation
            % Expected values
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            refCoefs = [];
            tmpArrays = cell(nVerSplit,nHorSplit,nDepSplit);
            for iCh = 1:nChs
                iSplit = 0;
                for iLay = 1:nDepSplit
                    for iCol = 1:nHorSplit
                        for iRow = 1:nVerSplit
                            iSplit = iSplit + 1;
                            tmpArrays{iRow,iCol,iLay} = subCoefArrays{iSplit,iCh};
                        end
                    end
                end
                tmpArray = cell2mat(tmpArrays);
                refCoefs = [ refCoefs tmpArray(:).'];
            end
            refScales = repmat([ height_ width_ depth_ ],[7*nLevels+1, 1]);
            imgExpctd = step(refSynthesizer,refCoefs,refScales);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel,...
                'InputType','Cell');
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,subScales);
            
            % Evaluation
            %testCase.assertFail('TODO: Check for split');
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testUdHaarSplittingCellInputLevel(testCase,level,useparallel)
            
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
            nChs = 7*level+1;
            nSplit = nVerSplit*nHorSplit*nDepSplit;
            subCoefArrays = cell(nSplit,nChs);
            subCoefs = cell(nSplit,1);
            for iSplit = 1:nSplit
                subCoefs{iSplit} = [];
                for iCh = 1:nChs
                    coefVec = rand(1,height_*width_*depth_/nSplit);
                    subCoefArrays{iSplit,iCh} = reshape(coefVec,...
                        [height_/nVerSplit width_/nHorSplit depth_/nDepSplit]);
                    subCoefs{iSplit} = [ subCoefs{iSplit} coefVec ];
                end
            end
            subScales = repmat([ height_/nVerSplit width_/nHorSplit depth_/nDepSplit],...
                [7*level+1, 1]);
            
            % Preparation
            % Expected values
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            refCoefs = [];
            tmpArrays = cell(nVerSplit,nHorSplit,nDepSplit);
            for iCh = 1:nChs
                iSplit = 0;
                for iLay = 1:nDepSplit
                    for iCol = 1:nHorSplit
                        for iRow = 1:nVerSplit
                            iSplit = iSplit + 1;
                            tmpArrays{iRow,iCol,iLay} = subCoefArrays{iSplit,iCh};
                        end
                    end
                end
                tmpArray = cell2mat(tmpArrays);
                refCoefs = [ refCoefs tmpArray(:).'];
            end
            refScales = repmat([ height_ width_ depth_ ],[7*level+1, 1]);
            imgExpctd = step(refSynthesizer,refCoefs,refScales);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel,...
                'InputType','Cell');
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,subScales);
            
            % Evaluation
            %testCase.assertFail('TODO: Check for split');
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testUdHaarSplittingWarningReconstruction(testCase,width,height,depth)
            
            % Parameters
            level_ = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(level_-1)-2;
            nHorPad = 2^(level_-1)-2;
            nDepPad = 2^(level_-1)-2;
            nChs = 7*level_+1;
            subCoefs = repmat(rand(1,height*width*depth),[1 nChs]);
            scales = repmat([ height width depth ],[7*level_+1, 1]);
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:ReconstructionFailureException';
            messageExpctd = 'Failure occurs in reconstruction. Please check the split and padding size.';
            
            % Preparation
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',false);
            
            % Actual values
            try
                step(testCase.synthesizer,subCoefs,scales);
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
            level_ = 2;
            nVerSplit = 2;
            nHorSplit = 2;
            nDepSplit = 2;
            nVerPad = 2^(level_-1)-2;
            nHorPad = 2^(level_-1)-2;
            nDepPad = 2^(level_-1)-2;
            nChs = 7*level_+1;
            subCoefs = repmat(rand(1,height*width*depth),[1 nChs]);
            scales = repmat([ height width depth ],[7*level_+1, 1]);
            
            % Preparation
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',false,...
                'IsIntegrityTest',false);
            
            % Evaluation
            try
                step(testCase.synthesizer,subCoefs,scales);
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
            nVerPad = 2^(level-1)-1;
            nHorPad = 2^(level-1)-1;
            nDepPad = 2^(level-1)-1;
            nChs = 7*level+1;
            subCoefs = repmat(rand(1,height*width*depth),[1 nChs]);
            scales = repmat([ height width depth],[7*level+1, 1]);
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalSplitFactorException';
            messageExpctd = 'Split factor must be a divisor of array size.';
            
            % Preparation
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer,...
                'VerticalSplitFactor',nVerSplit,...
                'HorizontalSplitFactor',nHorSplit,...
                'DepthSplitFactor',nDepSplit,...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',false);
            
            % Actual values
            try
                step(testCase.synthesizer,subCoefs,scales);
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
            nVerPad = 2^(level-1)-1;
            nHorPad = 2^(level-1)-1;
            nDepPad = 2^(level-1)-1;
            nChs = 7*level+1;
            subCoefs = repmat(rand(1,height*width*depth),[1 nChs]);
            scales = repmat([ height width depth],[7*level+1, 1]);
            
            % Preparation
            % Expected values
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            imgExpctd = step(refSynthesizer,subCoefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',useparallel);
            
            % Actual values
            imgActual = step(testCase.synthesizer,subCoefs,scales);
            
            % Evaluation
            %testCase.assertFail('TODO: Check for split');
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testUdHaarSplitFactorWarning(testCase,width,height,depth,level)
            
            % Parameters
            nVerSplit = 3;
            nHorSplit = 3;
            nDepSplit = 3;
            nVerPad = 2^(level-1)-1;
            nHorPad = 2^(level-1)-1;
            nDepPad = 2^(level-1)-1;
            nChs = 7*level+1;
            subCoefs = repmat(rand(1,height*width*depth),[1 nChs]);
            scales = repmat([ height width depth],[7*level+1, 1]);
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalSplitFactorException';
            messageExpctd = 'Split factor must be a divisor of array size.';
            
            % Preparation
            import saivdr.dictionary.udhaar.*
            refSynthesizer = UdHaarSynthesis3dSystem();
            
            % Instantiation of target class
            import saivdr.dictionary.olaols.*
            testCase.synthesizer = Synthesis3dOlaWrapper(...
                'Synthesizer',refSynthesizer,...
                'SplitFactor',[nVerSplit,nHorSplit,nDepSplit],...
                'PadSize',[nVerPad,nHorPad,nDepPad],...
                'UseParallel',false);
            
            % Actual values
            try
                step(testCase.synthesizer,subCoefs,scales);
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
end
