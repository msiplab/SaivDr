classdef NsoltSynthesis3dSystemTestCase < matlab.unittest.TestCase
    %NSOLTSYNTHESIS3DSYSTEMTESTCASE Test case for NsoltSynthesis3dSystem
    %
    % SVN identifier:
    % $Id: NsoltSynthesis3dSystemTestCase.m 869 2015-11-26 09:03:07Z sho $
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2017, Shogo MURAMATSU
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
            import saivdr.dictionary.nsoltx.*
            lppufbExpctd = OvsdLpPuFb3dTypeIVm1System(...
                'OutputMode','ParameterMatrixSet');
            frmbdExpctd  = 1;
            
            % Instantiation
            testCase.synthesizer = NsoltSynthesis3dSystem();
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb3d');
            frmbdActual  = get(testCase.synthesizer,'FrameBound');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
            testCase.assertEqual(frmbdActual,frmbdExpctd);
        end
        
        % Test
        function testDefaultConstruction4plus4(testCase)
            
            % Preperation
            nChs = [4 4];
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufbExpctd = OvsdLpPuFb3dTypeIVm1System(...
                'NumberOfChannels',nChs,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.ChannelGroup
            testCase.synthesizer = NsoltAnalysis3dSystem(...
                'NumberOfSymmetricChannels',nChs(ChannelGroup.UPPER),...
                'NumberOfAntisymmetricChannels',nChs(ChannelGroup.LOWER));
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb3d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end
        
        % Test for default construction
        function testInverseBlockDct(testCase)
            
            dec = 2;
            height = 16;
            width  = 16;
            depth  = 16;
            nChs   = dec*dec*dec;
            subCoefs  = rand(height*nChs/dec,width/dec,depth/dec);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec]);
            E0 = step(lppufb,[],[]);
            imgExpctd = zeros(height,width,depth);
            for iz = 1:depth/dec
                for ix = 1:width/dec
                    for iy = 1:height/dec
                        blockData = subCoefs((iy-1)*nChs+1:iy*nChs,ix,iz);
                        imgExpctd(...
                            (iy-1)*dec+1:iy*dec,...
                            (ix-1)*dec+1:ix*dec,...
                            (iz-1)*dec+1:iz*dec) = ...
                            reshape(flipud(E0.'*blockData),dec,dec,dec);
                    end
                end
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        %{
function testInverseBlockDctDec44(testCase)
            
            dec = 4;
            height = 32;
            width = 32;
            subCoefs  = rand(height*dec,width/dec);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iCh = 1:dec*dec
                subImg = subCoefs(iCh:dec*dec:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec]);
            E0 = step(lppufb,[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(subCoefs,[dec*dec 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        %}
        
        % Test
        function testStepDec222Ch44Ord000(testCase)
            dec = 2;
            nChs = dec*dec*dec;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = rand(height*nChs/dec,width/dec,depth/dec);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem();
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            imgExpctd = zeros(height,width,depth);
            for iz = 1:depth/dec
                for ix = 1:width/dec
                    for iy = 1:height/dec
                        blockData = subCoefs((iy-1)*nChs+1:iy*nChs,ix,iz);
                        imgExpctd(...
                            (iy-1)*dec+1:iy*dec,...
                            (ix-1)*dec+1:ix*dec,...
                            (iz-1)*dec+1:iz*dec) = ...
                            reshape(flipud(E.'*blockData),dec,dec,dec);
                    end
                end
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec222Ch44Ord002(testCase)
            
            dec = 2;
            ord = [ 0 0 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nChs = dec*dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord002PeriodicExt(testCase)
            
            dec = 2;
            ord = [ 0 0 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nDecs = dec*dec*dec;
            subCoefs = cell(nDecs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nDecs
                if iSubband == 1
                    subImg = randn(height/dec,width/dec,depth/dec);
                else
                    subImg = zeros(height/dec,width/dec,depth/dec);
                end
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nDecs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        
        % Test
        function testStepDec222Ch44Ord020(testCase)
            
            dec = 2;
            ord = [ 0 2 0 ];
            height = 16;
            width = 16;
            depth = 16;
            nChs = dec*dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord200PeriodicExt(testCase)
            
            dec = 2;
            ord = [ 2 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            nDecs = dec*dec*dec;
            subCoefs = cell(nDecs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nDecs
                if iSubband == 1
                    subImg = randn(height/dec,width/dec,depth/dec);
                else
                    subImg = zeros(height/dec,width/dec,depth/dec);
                end
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nDecs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord022(testCase)
            
            dec = 2;
            ord = [ 0 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nChs = dec*dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord022eriodicExt(testCase)
            
            dec = 2;
            ord = [ 0 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nDecs = dec*dec*dec;
            subCoefs = cell(nDecs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nDecs
                if iSubband == 1
                    subImg = randn(height/dec,width/dec,depth/dec);
                else
                    subImg = zeros(height/dec,width/dec,depth/dec);
                end
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nDecs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord202(testCase)
            
            dec = 2;
            ord = [ 2 0 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nChs = dec*dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord202eriodicExt(testCase)
            
            dec = 2;
            ord = [ 2 0 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nDecs = dec*dec*dec;
            subCoefs = cell(nDecs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nDecs
                if iSubband == 1
                    subImg = randn(height/dec,width/dec,depth/dec);
                else
                    subImg = zeros(height/dec,width/dec,depth/dec);
                end
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nDecs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord222(testCase)
            
            dec = 2;
            ord = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nChs = dec*dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord222PeriodicExt(testCase)
            
            dec = 2;
            ord = [ 2 2 2 ];
            height = 16;
            width = 32;
            depth = 64;
            nDecs = dec*dec*dec;
            subCoefs = cell(nDecs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nDecs
                if iSubband == 1
                    subImg = randn(height/dec,width/dec,depth/dec);
                else
                    subImg = zeros(height/dec,width/dec,depth/dec);
                end
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nDecs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
        end
        
        
        % Test
        function testStepDec222Ch66Ord222PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 6 6 ];
            ord = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nChs  = sum(chs);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                if iSubband == 1
                    subImg = randn(height/dec,width/dec,depth/dec);
                else
                    subImg = zeros(height/dec,width/dec,depth/dec);
                end
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,....
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord444(testCase)
            
            dec = 2;
            ord = 4;
            height = 16;
            width = 16;
            depth = 16;
            nDecs = dec*dec*dec;
            subCoefs = cell(nDecs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nDecs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nDecs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord444PeriodicExt(testCase)
            
            dec = 2;
            ord = 4;
            height = 16;
            width = 16;
            depth = 16;
            nDecs = dec*dec*dec;
            subCoefs = cell(nDecs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nDecs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nDecs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord666(testCase)
            
            dec = 2;
            ord = 6;
            height = 16;
            width = 16;
            depth = 16;
            nDecs = dec*dec*dec;
            subCoefs = cell(nDecs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nDecs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nDecs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(3*dec+1:end-3*dec,3*dec+1:end-3*dec,3*dec+1:end-3*dec); % ignore border
            imgActual = imgActual(3*dec+1:end-3*dec,3*dec+1:end-3*dec,3*dec+1:end-3*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord666PeriodicExt(testCase)
            
            dec = 2;
            ord = 6;
            height = 16;
            width = 16;
            depth = 16;
            nDecs = dec*dec*dec;
            subCoefs = cell(nDecs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nDecs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nDecs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        %{
        
function testStepDec222Ch666Ord00(testCase)
            
            dec = 2;
            ch = 6;
            ord = 0;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        %Dec22Ch8Ord00
function testStepDec22Ch8Ord00(testCase)
            
            dec = 2;
            ch = 8;
            ord = 0;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        %Dec11Ch4Ord22
function testStepDec11Ch4Ord22(testCase)
            
            dec = 1;
            ch = 4;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord22
function testStepDec22Ch4Ord22(testCase)
            
            dec = 2;
            ch = 4;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord22
function testStepDec22Ch6Ord22(testCase)
            
            dec = 2;
            ch  = 6;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord22
function testStepDec22Ch8Ord22(testCase)
            
            dec = 2;
            ch  = 8;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord44
function testStepDec11Ch4Ord44(testCase)
            
            dec = 1;
            ch  = 4;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord44
function testStepDec22Ch4Ord44(testCase)
            
            dec = 2;
            ch  = 4;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord44
function testStepDec22Ch6Ord44(testCase)
            
            dec = 2;
            ch  = 6;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord44
function testStepDec22Ch8Ord44(testCase)
            
            dec = 2;
            ch  = 8;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord66
function testStepDec11Ch4Ord66(testCase)
            
            dec = 1;
            ch  = 4;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord66
function testStepDec2Ch4Ord66(testCase)
            
            dec = 2;
            ch  = 4;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord66
function testStepDec22Ch6Ord66(testCase)
            
            dec = 2;
            ch  = 6;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord66
function testStepDec22Ch8Ord66(testCase)
            
            dec = 2;
            ch  = 8;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
                    
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord66PeriodicExt
function testStepDec11Ch4Ord66PeriodicExt(testCase)
            
            dec = 1;
            ch  = 4;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord66PeriodicExt
function testStepDec22Ch4Ord66PeriodicExt(testCase)
            
            dec = 2;
            ch  = 4;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord66PeriodicExt
function testStepDec22Ch6Ord66PeriodicExt(testCase)
            
            dec = 2;
            ch  = 6;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord66PeriodicExt
function testStepDec22Ch8Ord66PeriodicExt(testCase)
            
            dec = 2;
            ch  = 8;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord00Level1
function testStepDec11Ch4Ord00Level1(testCase)
            
            dec = 1;
            ch = 4;
            ord = 0;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
                        
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord00Level1
function testStepDec22Ch4Ord00Level1(testCase)
            
            dec = 2;
            ch = 4;
            ord = 0;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord00Level1
function testStepDec22Ch6Ord00Level1(testCase)
            
            dec = 2;
            ch = 6;
            ord = 0;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
             coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord00Level1
function testStepDec22Ch8Ord00Level1(testCase)
            
            dec = 2;
            ch = 8;
            ord = 0;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord00Level2
function testStepDec11Ch4Ord00Level2(testCase)
            
            dec = 1;
            ch = 4;
            ord = 0;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 0; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord00Level2
function testStepDec22Ch4Ord00Level2(testCase)
            
            dec = 2;
            ch = 4;
            ord = 0;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord00Level2
function testStepDec22Ch6Ord00Level2(testCase)
            
            dec = 2;
            ch = 6;
            ord = 0;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord00Level2
function testStepDec22Ch8Ord00Level2(testCase)
            
            dec = 2;
            ch = 8;
            ord = 0;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2));
            subCoefs{8} = rand(height/(dec^2),width/(dec^2));
            subCoefs{9} = rand(height/(dec),width/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord22Level1
function testStepDec11Ch4Ord22Level1(testCase)
            
            dec = 1;
            ch = 4;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord22Level1
function testStepDec22Ch4Ord22Level1(testCase)
            
            dec = 2;
            ch = 4;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord22Level1
function testStepDec22Ch6Ord22Level1(testCase)
            
            dec = 2;
            ch = 6;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord22Level1
function testStepDec22Ch8Ord22Level1(testCase)
            
            dec = 2;
            ch = 8;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord22Level2PeriodicExt
function testStepDec11Ch4Ord22Level2PeriodicExt(testCase)
            
            dec = 1;
            ch = 4;
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 0; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord22Level2PeriodicExt
function testStepDec22Ch4Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 4;
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord22Level2PeriodicExt
function testStepDec22Ch6Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 6;
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch44Ord22Level2PeriodicExt
function testStepDec22Ch44Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 4 4 ];
            ch = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2));
            subCoefs{8} = rand(height/(dec^2),width/(dec^2));
            subCoefs{9} = rand(height/(dec),width/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch44Ord44Level3PeriodicExt
function testStepDec22Ch44Ord44Level3PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 4 4 ];
            ch = sum(decch(3:4));
            ord = 4;
            height = 64;
            width = 64;
            nLevels = 3;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^3),width/(dec^3));
            subCoefs{2} = rand(height/(dec^3),width/(dec^3));
            subCoefs{3} = rand(height/(dec^3),width/(dec^3));
            subCoefs{4} = rand(height/(dec^3),width/(dec^3));
            subCoefs{5} = rand(height/(dec^3),width/(dec^3));
            subCoefs{6} = rand(height/(dec^3),width/(dec^3));
            subCoefs{7} = rand(height/(dec^3),width/(dec^3));
            subCoefs{8} = rand(height/(dec^3),width/(dec^3));
            subCoefs{9} = rand(height/(dec^2),width/(dec^2));
            subCoefs{10} = rand(height/(dec^2),width/(dec^2));
            subCoefs{11} = rand(height/(dec^2),width/(dec^2));
            subCoefs{12} = rand(height/(dec^2),width/(dec^2));
            subCoefs{13} = rand(height/(dec^2),width/(dec^2));
            subCoefs{14} = rand(height/(dec^2),width/(dec^2));
            subCoefs{15} = rand(height/(dec^2),width/(dec^2));
            subCoefs{16} = rand(height/(dec),width/(dec));
            subCoefs{17} = rand(height/(dec),width/(dec));
            subCoefs{18} = rand(height/(dec),width/(dec));
            subCoefs{19} = rand(height/(dec),width/(dec));
            subCoefs{20} = rand(height/(dec),width/(dec));
            subCoefs{21} = rand(height/(dec),width/(dec));
            subCoefs{22} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
function testSetLpPuFb3dDec22Ch44Ord44(testCase)
            
            dec = 2;
            decch = [ dec dec 4 4 ];
            ord = 4;
            height = 32;
            width = 32;
            subCoefs{1} = rand(height/(dec),width/(dec));
            subCoefs{2} = rand(height/(dec),width/(dec));
            subCoefs{3} = rand(height/(dec),width/(dec));
            subCoefs{4} = rand(height/(dec),width/(dec));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Termination');
            imgPre = step(testCase.synthesizer,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyTrue(diff<1e-15);
                        
            % ReInstantiation of target class
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
function testIsCloneLpPuFb3dFalse(testCase)
            
            dec = 2;
            ch = [ 4 4 ];
            ord = 4;
            height = 32;
            width = 32;
            subCoefs{1} = rand(height/(dec),width/(dec));
            subCoefs{2} = rand(height/(dec),width/(dec));
            subCoefs{3} = rand(height/(dec),width/(dec));
            subCoefs{4} = rand(height/(dec),width/(dec));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'IsCloneLpPuFb3d',true);
            
            % Pre
            imgPre = step(testCase.synthesizer,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Pst
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,sprintf('%g',diff));
            
            % ReInstantiation of target class
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'IsCloneLpPuFb3d',false);
            
            % Pre
            imgPre = step(testCase.synthesizer,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Pst
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end
        
        % Test
function testClone(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 4 4 ];
            ord = [ 4 4 ];
            height = 64;
            width  = 64;
            coefs = rand(sum(ch)/prod(dec)*height*width,1);
            scales = repmat([height/dec(1) width/dec(2)],[sum(ch) 1]);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Termination');
            
            % Clone
            cloneSynthesizer = clone(testCase.synthesizer);
            
            % Evaluation
            testCase.verifyEqual(cloneSynthesizer,testCase.synthesizer);
            testCase.verifyFalse(cloneSynthesizer == testCase.synthesizer);
            prpOrg = get(testCase.synthesizer,'LpPuFb3d');
            prpCln = get(cloneSynthesizer,'LpPuFb3d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            recImgExpctd = step(testCase.synthesizer,coefs,scales);
            recImgActual = step(cloneSynthesizer,coefs,scales);
            testCase.assertEqual(recImgActual,recImgExpctd);
            
        end
        %}
        % Test
        function testConstructionTypeII(testCase)
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufbExpctd = OvsdLpPuFb3dTypeIIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufbExpctd);
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb3d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end
        
        % Test
        function testStepDec222Ch54Ord222PeriodicExt(testCase)
            
            dec = 2;
            nch = [ 5 4 ];
            ord = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nChs = sum(nch);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                if iSubband == 1
                    subImg = randn(height/dec,width/dec,depth/dec);
                else
                    subImg = zeros(height/dec,width/dec,depth/dec);
                end
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', nch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch64Ord222PeriodicExt(testCase)
            
            dec = 2;
            nch = [ 6 4 ];
            ord = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nChs = sum(nch);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                if iSubband == 1
                    subImg = randn(height/dec,width/dec,depth/dec);
                else
                    subImg = zeros(height/dec,width/dec,depth/dec);
                end
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', nch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testDefaultConstruction6plus4(testCase)
            
            % Preperation
            nChs = [6 4];
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufbExpctd = OvsdLpPuFb3dTypeIIVm1System(...
                'NumberOfChannels',nChs,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.ChannelGroup
            testCase.synthesizer = NsoltAnalysis3dSystem(...
                'NumberOfSymmetricChannels',nChs(ChannelGroup.UPPER),...
                'NumberOfAntisymmetricChannels',nChs(ChannelGroup.LOWER));
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb3d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end
        
        % Test
        function testStepDec222Ch54Ord222Vm0(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = 2;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            import saivdr.dictionary.nsoltx.*
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        function testStepDec222Ch54Ord222Vm1(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = 2;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            import saivdr.dictionary.nsoltx.*
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch54Ord444PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = 4;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch54Ord666PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = 6;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch54Ord002(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = [ 0 0 2 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,....
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,:,dec+1:end-dec); % ignore border
            imgActual = imgActual(:,:,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch54Ord020(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = [ 0 2 0 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,dec+1:end-dec,:); % ignore border
            imgActual = imgActual(:,dec+1:end-dec,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch54Ord200(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = [ 2 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,:,:); % ignore border
            imgActual = imgActual(dec+1:end-dec,:,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch54Ord222(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = [ 2 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch54Ord022(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = [ 0 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(:,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch54Ord202(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = [ 2 0 2 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,:,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,:,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch54Ord220(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = [ 2 2 0 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,:); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch54Ord444(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = [ 4 4 4 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch64Ord444(testCase)
            
            dec = 2;
            chs = [ 6 4 ];
            nChs = sum(chs);
            ord = [ 4 4 4 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch64Ord444PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 6 4 ];
            nChs = sum(chs);
            ord = [ 4 4 4 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec111Ch22Ord000(testCase)
            
            dec = 1;
            chs = [ 2 2 ];
            nChs = sum(chs);
            ord = 0;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(subCoefs{iSubband},atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch5Ord00
        function testStepDec111Ch32Ord000(testCase)
            
            dec = 1;
            chs = [ 3 2 ];
            nChs = sum(chs);
            ord = 0;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(subCoefs{iSubband},atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec111Ch22Ord222(testCase)
            
            dec = 1;
            chs = [ 2 2 ];
            nChs = sum(chs);
            ord = 2;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(subCoefs{iSubband},atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch5Ord00
        function testStepDec111Ch32Ord222(testCase)
            
            dec = 1;
            chs = [ 3 2 ];
            nChs = sum(chs);
            ord = 2;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(subCoefs{iSubband},atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec222Ch54Ord000(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end

        %Test
        function testStepDec222Ch44Ord000Level2(testCase)
            
            dec = 2;
            chs = [ 4 4 ];
            nChs = sum(chs);
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{8} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{9} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = subCoefs{1};
            for iLevel = 1:nLevels
                atom = step(lppufb,[],[],1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs,dec,phase),atom,'cir');
                for iSubSub = 2:nChs
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    atom = step(lppufb,[],[],iSubSub);
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec222Ch44Ord222Level2PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 4 4 ];
            nChs = sum(chs);
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{8} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{9} = rand(height/(dec),width/(dec),depth/(dec));            
            subCoefs{10} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = subCoefs{1};
            for iLevel = 1:nLevels
                atom = step(lppufb,[],[],1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs,dec,phase),atom,'cir');
                for iSubSub = 2:nChs
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    atom = step(lppufb,[],[],iSubSub);
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec222Ch44Ord222Level3PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 4 4 ];
            nChs = sum(chs);
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            nLevels = 3;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{2} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{3} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{4} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{5} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{6} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{7} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{8} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{9} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{10} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{11} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{12} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{13} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{14} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{15} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{16} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{17} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{18} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{19} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{20} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{21} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{22} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);            
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = subCoefs{1};
            for iLevel = 1:nLevels
                atom = step(lppufb,[],[],1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs,dec,phase),atom,'cir');
                for iSubSub = 2:nChs
                    atom = step(lppufb,[],[],iSubSub);
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec222Ch66Ord444Level3PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 6 6 ];
            nChs = sum(chs);
            ord = 4;
            height = 32;
            width = 32;
            depth = 32;
            nLevels = 3;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{2} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{3} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{4} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{5} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{6} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{7} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{8} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{9} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{10} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));        
            subCoefs{11} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{12} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{13} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{14} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{15} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{16} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{17} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{18} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{19} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{20} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{21} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{22} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{23} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));            
            subCoefs{24} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{25} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{26} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{27} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{28} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{29} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{30} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{31} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{32} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{33} = rand(height/(dec),width/(dec),depth/(dec));            
            subCoefs{34} = rand(height/(dec),width/(dec),depth/(dec));                        
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);            
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = subCoefs{1};
            for iLevel = 1:nLevels
                atom = step(lppufb,[],[],1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs,dec,phase),atom,'cir');
                for iSubSub = 2:nChs
                    atom = step(lppufb,[],[],iSubSub);
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        
        
        
        %Test
        function testStepDec222Ch54Ord000Level2(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{8} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{9} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{10} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{16} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{17} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = subCoefs{1};
            for iLevel = 1:nLevels
                atom = step(lppufb,[],[],1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs,dec,phase),atom,'cir');
                for iSubSub = 2:nChs
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    atom = step(lppufb,[],[],iSubSub);
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec222Ch54Ord222Level2PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{8} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{9} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{10} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{16} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{17} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = subCoefs{1};
            for iLevel = 1:nLevels
                atom = step(lppufb,[],[],1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs,dec,phase),atom,'cir');
                for iSubSub = 2:nChs
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    atom = step(lppufb,[],[],iSubSub);
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec222Ch54Ord222Level3PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 5 4 ];
            nChs = sum(chs);
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            nLevels = 3;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{2} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{3} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{4} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{5} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{6} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{7} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{8} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{9} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{10} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{11} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{12} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{13} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{14} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{15} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{16} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{17} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{18} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{19} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{20} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{21} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{22} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{23} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{24} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{25} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);            
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = subCoefs{1};
            for iLevel = 1:nLevels
                atom = step(lppufb,[],[],1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs,dec,phase),atom,'cir');
                for iSubSub = 2:nChs
                    atom = step(lppufb,[],[],iSubSub);
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec222Ch64Ord444Level3PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 6 4 ];
            nChs = sum(chs);
            ord = 4;
            height = 32;
            width = 32;
            depth = 32;
            nLevels = 3;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{2} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{3} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{4} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{5} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{6} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{7} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{8} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{9} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{10} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));        
            subCoefs{11} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{12} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{13} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{14} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{15} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{16} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{17} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{18} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{19} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{20} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{21} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{22} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{23} = rand(height/(dec),width/(dec),depth/(dec));        
            subCoefs{24} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{25} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{26} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{27} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{28} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);            
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = subCoefs{1};
            for iLevel = 1:nLevels
                atom = step(lppufb,[],[],1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs,dec,phase),atom,'cir');
                for iSubSub = 2:nChs
                    atom = step(lppufb,[],[],iSubSub);
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test        
        function testSetLpPuFb3dDec222Ch64Ord444(testCase)
            
            dec = 2;
            ch = [ 6 4 ];
            ord = 4;
            height = 32;
            width = 32;
            depth = 32;
            subCoefs{1} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{2} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{3} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{4} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{5} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord]);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            imgPre = step(testCase.synthesizer,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyTrue(diff<1e-15);
            
            % ReInstantiation of target class
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        %}
        
        %Test
        function testSetLpPuFb3dDec222Ch54Ord444(testCase)
            
            dec = 2;
            ch = [ 5 4 ];
            ord = 4;
            height = 32;
            width = 32;
            depth = 32;
            subCoefs{1} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{2} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{3} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{4} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{5} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord]);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            imgPre = step(testCase.synthesizer,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyTrue(diff<1e-15);
            
            % ReInstantiation of target class
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
        % Test
        function testCloneTypeII(testCase)
            
            dec = [ 2 2 2 ];
            ch =  [ 5 4 ];
            ord = [ 4 4 4 ];
            height = 64;
            width  = 64;
            depth  = 64;
            coefs = rand(sum(ch)/prod(dec)*height*width*depth,1);
            scales = repmat([height/dec(1) width/dec(2) depth/dec(3)],[sum(ch) 1]);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Termination');
            
            % Clone
            cloneSynthesizer = clone(testCase.synthesizer);
            
            % Evaluation
            testCase.verifyEqual(cloneSynthesizer,testCase.synthesizer);
            testCase.verifyFalse(cloneSynthesizer == testCase.synthesizer);
            prpOrg = get(testCase.synthesizer,'LpPuFb3d');
            prpCln = get(cloneSynthesizer,'LpPuFb3d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            recImgExpctd = step(testCase.synthesizer,coefs,scales);
            recImgActual = step(cloneSynthesizer,coefs,scales);
            testCase.assertEqual(recImgActual,recImgExpctd);
            
        end
        
        %{
        function testDifferetiationTypeI(testCase)
            
            dec = [ 2 2 2 ];
            ch =  [ 5 5 ];
            ord = [ 0 0 0 ];
            height = 16;
            width  = 16;
            depth  = 16;
            iAng   = 1;
            nChs   = sum(ch);
            subCoefs  = rand(height*nChs/dec(1),width/dec(2),depth/dec(3));
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'IsDifferentiation',true,...
                'BoundaryOperation','Termination');
            %s = matlab.System.saveObject(testCase.analyzer);
            
            % Actual values
            set(testCase.synthesizer,'IndexOfDifferentiationAngle',iAng);
            
            recImg = step(testCase.synthesizer,coefs,scales);

            % Evaluation
            %testCase.verifyEqual(scalesActual,scalesExpctd);
            %diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            %testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
            %    sprintf('%g',diff));
        end
        %}
        %{
        % Test
        function testStepDec444Ch3232Ord000(testCase)
            
            dec = 4;
            chs = [ 32 32 ];
            nChs = sum(chs);
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = rand(height*nChs/dec,width/dec,depth/dec);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            imgExpctd = zeros(height,width,depth);
            for iz = 1:depth/dec
                for ix = 1:width/dec
                    for iy = 1:height/dec
                        blockData = subCoefs((iy-1)*nChs+1:iy*nChs,ix,iz);
                        imgExpctd(...
                            (iy-1)*dec+1:iy*dec,...
                            (ix-1)*dec+1:ix*dec,...
                            (iz-1)*dec+1:iz*dec) = ...
                            reshape(flipud(E.'*blockData),dec,dec,dec);
                    end
                end
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        %}
         
        % Test
        function testStepDec222Ch45Ord222PeriodicExt(testCase)
            
            dec = 2;
            nch = [ 4 5 ];
            ord = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nChs = sum(nch);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                if iSubband == 1
                    subImg = randn(height/dec,width/dec,depth/dec);
                else
                    subImg = zeros(height/dec,width/dec,depth/dec);
                end
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', nch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch46Ord222PeriodicExt(testCase)
            
            dec = 2;
            nch = [ 4 6 ];
            ord = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nChs = sum(nch);
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                if iSubband == 1
                    subImg = randn(height/dec,width/dec,depth/dec);
                else
                    subImg = zeros(height/dec,width/dec,depth/dec);
                end
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', nch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testDefaultConstruction4plus6(testCase)
            
            % Preperation
            nChs = [ 4 6 ];
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufbExpctd = OvsdLpPuFb3dTypeIIVm1System(...
                'NumberOfChannels',nChs,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.ChannelGroup
            testCase.synthesizer = NsoltAnalysis3dSystem(...
                'NumberOfSymmetricChannels',nChs(ChannelGroup.UPPER),...
                'NumberOfAntisymmetricChannels',nChs(ChannelGroup.LOWER));
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb3d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end
        
        % Test
        function testStepDec222Ch45Ord222Vm0(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = 2;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            import saivdr.dictionary.nsoltx.*
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        function testStepDec222Ch45Ord222Vm1(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = 2;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            import saivdr.dictionary.nsoltx.*
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch45Ord444PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = 4;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch45Ord666PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = 6;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch45Ord002(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = [ 0 0 2 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,....
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,:,dec+1:end-dec); % ignore border
            imgActual = imgActual(:,:,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch45Ord020(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = [ 0 2 0 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,dec+1:end-dec,:); % ignore border
            imgActual = imgActual(:,dec+1:end-dec,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch45Ord200(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = [ 2 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,:,:); % ignore border
            imgActual = imgActual(dec+1:end-dec,:,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch45Ord222(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = [ 2 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch45Ord022(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = [ 0 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(:,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch45Ord202(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = [ 2 0 2 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,:,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,:,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch45Ord220(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = [ 2 2 0 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,:); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch45Ord444(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = [ 4 4 4 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch46Ord444(testCase)
            
            dec = 2;
            chs = [ 4 6 ];
            nChs = sum(chs);
            ord = [ 4 4 4 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch46Ord444PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 4 6 ];
            nChs = sum(chs);
            ord = [ 4 4 4 ];
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(dec*dec*dec,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end

        %Dec11Ch5Ord00
        function testStepDec111Ch23Ord000(testCase)
            
            dec = 1;
            chs = [ 2 3 ];
            nChs = sum(chs);
            ord = 0;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(subCoefs{iSubband},atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch5Ord00
        function testStepDec111Ch23Ord222(testCase)
            
            dec = 1;
            chs = [ 2 3 ];
            nChs = sum(chs);
            ord = 2;
            height = 16;
            width = 16;
            depth = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(subCoefs{iSubband},atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(....
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec222Ch45Ord000(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nChs,3);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec,depth/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(...
                    upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
    
        %Test
        function testStepDec222Ch45Ord000Level2(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{8} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{9} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{10} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{16} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{17} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = subCoefs{1};
            for iLevel = 1:nLevels
                atom = step(lppufb,[],[],1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs,dec,phase),atom,'cir');
                for iSubSub = 2:nChs
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    atom = step(lppufb,[],[],iSubSub);
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec222Ch45Ord222Level2PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{8} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{9} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{10} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{16} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{17} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = subCoefs{1};
            for iLevel = 1:nLevels
                atom = step(lppufb,[],[],1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs,dec,phase),atom,'cir');
                for iSubSub = 2:nChs
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    atom = step(lppufb,[],[],iSubSub);
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec222Ch45Ord222Level3PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 4 5 ];
            nChs = sum(chs);
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            nLevels = 3;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{2} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{3} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{4} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{5} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{6} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{7} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{8} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{9} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{10} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{11} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{12} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{13} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{14} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{15} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{16} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{17} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{18} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{19} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{20} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{21} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{22} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{23} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{24} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{25} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);            
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = subCoefs{1};
            for iLevel = 1:nLevels
                atom = step(lppufb,[],[],1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs,dec,phase),atom,'cir');
                for iSubSub = 2:nChs
                    atom = step(lppufb,[],[],iSubSub);
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec222Ch46Ord444Level3PeriodicExt(testCase)
            
            dec = 2;
            chs = [ 4 6 ];
            nChs = sum(chs);
            ord = 4;
            height = 32;
            width = 32;
            depth = 32;
            nLevels = 3;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{2} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{3} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{4} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{5} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{6} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{7} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{8} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{9} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));
            subCoefs{10} = rand(height/(dec^3),width/(dec^3),depth/(dec^3));        
            subCoefs{11} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{12} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{13} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{14} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{15} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{16} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{17} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{18} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{19} = rand(height/(dec^2),width/(dec^2),depth/(dec^2));
            subCoefs{20} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{21} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{22} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{23} = rand(height/(dec),width/(dec),depth/(dec));        
            subCoefs{24} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{25} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{26} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{27} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{28} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',chs,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,u,p),1),u,p),1),u,p),1);            
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = subCoefs{1};
            for iLevel = 1:nLevels
                atom = step(lppufb,[],[],1);
                imgExpctd = imfilter(...
                    upsample3_(subsubCoefs,dec,phase),atom,'cir');
                for iSubSub = 2:nChs
                    atom = step(lppufb,[],[],iSubSub);
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample3_(subCoefs{iSubband},dec,phase),atom,'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test        
        function testSetLpPuFb3dDec222Ch46Ord444(testCase)
            
            dec = 2;
            ch = [ 4 6 ];
            ord = 4;
            height = 32;
            width = 32;
            depth = 32;
            subCoefs{1} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{2} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{3} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{4} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{5} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord]);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            imgPre = step(testCase.synthesizer,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyTrue(diff<1e-15);
            
            % ReInstantiation of target class
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
        %Test
        function testSetLpPuFb3dDec222Ch45Ord444(testCase)
            
            dec = 2;
            ch = [ 4 5 ];
            ord = 4;
            height = 32;
            width = 32;
            depth = 32;
            subCoefs{1} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{2} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{3} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{4} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{5} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec),depth/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec),depth/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord]);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            imgPre = step(testCase.synthesizer,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyTrue(diff<1e-15);
            
            % ReInstantiation of target class
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
        % Test
        function testStepDec112Ch22Ord000(testCase)
            
            nDecs = [ 1 1 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 0 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end

        % Test
        function testStepDec121Ch22Ord000(testCase)
            
            nDecs = [ 1 2 1 ];
            nChs  = [ 2 2 ];
            nOrds = [ 0 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec211Ch22Ord000(testCase)
            
            nDecs = [ 2 1 1 ];
            nChs  = [ 2 2 ];
            nOrds = [ 0 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec112Ch22Ord222(testCase)
            
            nDecs = [ 1 1 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end

        % Test
        function testStepDec121Ch22Ord222(testCase)
            
            nDecs = [ 1 2 1 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec211Ch22Ord222(testCase)
            
            nDecs = [ 2 1 1 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec112Ch23Ord000(testCase)
            
            nDecs = [ 1 1 2 ];
            nChs  = [ 2 3 ];
            nOrds = [ 0 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end

        % Test
        function testStepDec121Ch23Ord000(testCase)
            
            nDecs = [ 1 2 1 ];
            nChs  = [ 2 3 ];
            nOrds = [ 0 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec211Ch23Ord000(testCase)
            
            nDecs = [ 2 1 1 ];
            nChs  = [ 2 3 ];
            nOrds = [ 0 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec112Ch23Ord222(testCase)
            
            nDecs = [ 1 1 2 ];
            nChs  = [ 2 3 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end

        % Test
        function testStepDec121Ch23Ord222(testCase)
            
            nDecs = [ 1 2 1 ];
            nChs  = [ 2 3 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec211Ch23Ord222(testCase)
            
            nDecs = [ 2 1 1 ];
            nChs  = [ 2 3 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        
         % Test
        function testStepDec112Ch32Ord000(testCase)
            
            nDecs = [ 1 1 2 ];
            nChs  = [ 3 2 ];
            nOrds = [ 0 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end

        % Test
        function testStepDec121Ch32Ord000(testCase)
            
            nDecs = [ 1 2 1 ];
            nChs  = [ 3 2 ];
            nOrds = [ 0 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec211Ch32Ord000(testCase)
            
            nDecs = [ 2 1 1 ];
            nChs  = [ 3 2 ];
            nOrds = [ 0 0 0 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec112Ch32Ord222(testCase)
            
            nDecs = [ 1 1 2 ];
            nChs  = [ 3 2 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end

        % Test
        function testStepDec121Ch32Ord222(testCase)
            
            nDecs = [ 1 2 1 ];
            nChs  = [ 3 2 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec211Ch32Ord222(testCase)
            
            nDecs = [ 2 1 1 ];
            nChs  = [ 3 2 ];
            nOrds = [ 2 2 2 ];
            height = 16;
            width = 16;
            depth = 16;
            nch_ = sum(nChs);
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width*depth);
            scales = zeros(nch_,3);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2),depth/nDecs(3));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            upsample3_ = @(x,u,p) ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,nDecs(1),p(1)),1),nDecs(2),p(2)),1),nDecs(3),p(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width,depth);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                atom = step(lppufb,[],[],iSubband);
                subbandImg = imfilter(upsample3_(...
                    subCoefs{iSubband},nDecs,phase),atom,'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb);
            
            % Actual values
            imgActual = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            imgActual = imgActual(nDecs(1)+1:end-nDecs(1),nDecs(2)+1:end-nDecs(2),nDecs(3)+1:end-nDecs(3)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
    end
    
end

