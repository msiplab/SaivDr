classdef TypeIISynthesisSystemTestCase < matlab.unittest.TestCase
    %TYPEIISYNTHESISSYSTEMTESTCASE Test case for TypeIISynthesisSystem
    %
    % SVN identifier:
    % $Id: TypeIISynthesisSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
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
            import saivdr.dictionary.nsolt.*
            lppufbExpctd = OvsdLpPuFb2dTypeIIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            testCase.synthesizer = TypeIISynthesisSystem();
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb2d');
            
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
            testCase.synthesizer = TypeIIAnalysisSystem(...
                'NumberOfSymmetricChannels',nChs(ChannelGroup.UPPER),...
                'NumberOfAntisymmetricChannels',nChs(ChannelGroup.LOWER));
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb2d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end
        % Test for default construction
        function testInverseBlockDctDec33(testCase)
            
            dec = 3;
            height = 24;
            width = 24;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]);
            E0 = step(lppufb,[],[]);
            %fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            %imgExpctd = blockproc(subCoefs,[dec*dec 1],fun);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(subCoefs,[dec*dec 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testInverseBlockDctDec55(testCase)
            
            dec = 5;
            height = 40;
            width = 40;
            subCoefs = rand(height*dec,width/dec);
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]);
            E0 = step(lppufb,[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(subCoefs,[dec*dec 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord00(testCase)
            
            dec = 2;
            nChs = 5;
            height = 16;
            width = 16;
            subCoefs = rand(height*nChs/dec,width/dec);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            fun = @(x) reshape(flipud(E.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(subCoefs,[nChs 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
            
        end
        
        % Test
        function testSynthesizerTypeIIDec44Cg16Ord00(testCase)
            
            dec = 4;
            nChs = 17;
            height = 32;
            width = 32;
            subCoefs = rand(height*nChs,width/dec);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            fun = @(x) reshape(flipud(E.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(subCoefs,[nChs 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord22Vm0(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            import saivdr.dictionary.nsolt.*
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord22Vm1(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            import saivdr.dictionary.nsolt.*
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord22PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord22(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord22PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctDec33Ord22(testCase)
            
            dec = 3;
            ord = 2;
            height = 24;
            width = 24;
            nChs = dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'PolyPhaseOrder',[ord ord]);
            
            
            % Expected values
            coefsExpctd = zeros(height,width);
            for iCol = 1:dec
                for iRow = 1:dec
                    iSubband = (iCol-1)*dec + iRow;
                    coefsExpctd = coefsExpctd + upsample(...
                        upsample(...
                        subCoefs{iSubband}.',dec,iCol-1).',dec,iRow-1);
                end
            end
            E0 = step(NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]),[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(coefsExpctd,...
                [dec dec],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctDec55Ord22(testCase)
            
            dec = 5;
            ord = 2;
            height = 40;
            width = 40;
            nChs = dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'PolyPhaseOrder',[ord ord]);
            
            
            % Expected values
            coefsExpctd = zeros(height,width);
            for iCol = 1:dec
                for iRow = 1:dec
                    iSubband = (iCol-1)*dec + iRow;
                    coefsExpctd = coefsExpctd + upsample(...
                        upsample(...
                        subCoefs{iSubband}.',dec,iCol-1).',dec,iRow-1);
                end
            end
            E0 = step(NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]),[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(coefsExpctd,...
                [dec dec],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord44(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord44PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec4Ch17Ord44(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord44PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testInverseBlockDctDec33Ord44(testCase)
            
            dec = 3;
            ord = 4;
            height = 24;
            width = 24;
            nChs = dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'PolyPhaseOrder',[ord ord]);
            
            
            % Expected values
            coefsExpctd = zeros(height,width);
            for iCol = 1:dec
                for iRow = 1:dec
                    iSubband = (iCol-1)*dec + iRow;
                    coefsExpctd = coefsExpctd + upsample(...
                        upsample(...
                        subCoefs{iSubband}.',dec,iCol-1).',dec,iRow-1);
                end
            end
            E0 = step(NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]),[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(coefsExpctd,...
                [dec dec],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctDec55Ord44(testCase)
            
            dec = 5;
            ord = 4;
            height = 40;
            width = 40;
            nChs = dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'PolyPhaseOrder',[ord ord]);
            
            
            % Expected values
            coefsExpctd = zeros(height,width);
            for iCol = 1:dec
                for iRow = 1:dec
                    iSubband = (iCol-1)*dec + iRow;
                    coefsExpctd = coefsExpctd + upsample(...
                        upsample(...
                        subCoefs{iSubband}.',dec,iCol-1).',dec,iRow-1);
                end
            end
            E0 = step(NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]),[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(coefsExpctd,...
                [dec dec],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord66(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(3*dec+1:end-3*dec,3*dec+1:end-3*dec); % ignore border
            imgActual = imgActual(3*dec+1:end-3*dec,3*dec+1:end-3*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIOrdDec22Ch566PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...'
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord66(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 6;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(3*dec+1:end-3*dec,3*dec+1:end-3*dec); % ignore border
            imgActual = imgActual(3*dec+1:end-3*dec,3*dec+1:end-3*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord66PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 6;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testInverseBlockDctDec33Ord66(testCase)
            
            dec = 3;
            ord = 6;
            height = 24;
            width = 24;
            nChs = dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            
            
            % Expected values
            coefsExpctd = zeros(height,width);
            for iCol = 1:dec
                for iRow = 1:dec
                    iSubband = (iCol-1)*dec + iRow;
                    coefsExpctd = coefsExpctd + upsample(...
                        upsample(...
                        subCoefs{iSubband}.',dec,iCol-1).',dec,iRow-1);
                end
            end
            E0 = step(NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]),[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(coefsExpctd,...
                [dec dec],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctDec55Ord66(testCase)
            
            dec = 5;
            ord = 6;
            height = 40;
            width = 40;
            nChs = dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'PolyPhaseOrder',[ord ord]);
            
            
            % Expected values
            coefsExpctd = zeros(height,width);
            for iCol = 1:dec
                for iRow = 1:dec
                    iSubband = (iCol-1)*dec + iRow;
                    coefsExpctd = coefsExpctd + upsample(...
                        upsample(...
                        subCoefs{iSubband}.',dec,iCol-1).',dec,iRow-1);
                end
            end
            E0 = step(NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]),[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(coefsExpctd,...
                [dec dec],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-8,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord02(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,dec+1:end-dec); % ignore border
            imgActual = imgActual(:,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord02(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,dec+1:end-dec); % ignore border
            imgActual = imgActual(:,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord04(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(:,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord04(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(:,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord20(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,:); % ignore border
            imgActual = imgActual(dec+1:end-dec,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord20(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,:); % ignore border
            imgActual = imgActual(dec+1:end-dec,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord40(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,:); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord40(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,:); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord02PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord02PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord04PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord04PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord20PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord20PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        
        %Test
        function testSynthesizerTypeIIDec22Ch5Ord40PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord40PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord24(testCase)
            
            dec = 2;
            nChs = 5;
            ord = [2 4];
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord24(testCase)
            
            dec = 4;
            nChs = 17;
            ord = [2 4];
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord24PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = [2 4];
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord24PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = [2 4];
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord42(testCase)
            
            dec = 2;
            nChs = 5;
            ord = [4 2];
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord42(testCase)
            
            dec = 4;
            nChs = 17;
            ord = [4 2];
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch5Ord42PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = [4 2];
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec44Ch17Ord42PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = [4 2];
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec11Ch5Ord00
        function testSynthesizerTypeIIDec11Ch5Ord00(testCase)
            
            dec = 1;
            ch = 5;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch7Ord00
        function testSynthesizerTypeIIDec22Ch7Ord00(testCase)
            
            dec = 2;
            ch = 7;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        %Dec22Ch9Ord00
        function testSynthesizerTypeIIDec22Ch9Ord00(testCase)
            
            dec = 2;
            ch = 9;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        %Dec22Ch11Ord00
        function testSynthesizerTypeIIDec22Ch11Ord00(testCase)
            
            dec = 2;
            ch = 11;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        %Dec11Ch4Ord22
        function testSynthesizerTypeIIDec11Ch5Ord22(testCase)
            
            dec = 1;
            ch = 5;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord22
        function testSynthesizerTypeIIDec22Ch7Ord22(testCase)
            
            dec = 2;
            ch = 7;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord22
        function testSynthesizerTypeIIDec22Ch9Ord22(testCase)
            
            dec = 2;
            ch  = 9;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord22
        function testSynthesizerTypeIIDec22Ch11Ord22(testCase)
            
            dec = 2;
            ch  = 11;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord44
        function testSynthesizerTypeIIDec11Ch5Ord44(testCase)
            
            dec = 1;
            ch  = 5;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord44
        function testSynthesizerTypeIIDec22Ch7Ord44(testCase)
            
            dec = 2;
            ch  = 7;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord44
        function testSynthesizerTypeIIDec22Ch9Ord44(testCase)
            
            dec = 2;
            ch  = 9;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord44
        function testSynthesizerTypeIIDec22Ch11Ord44(testCase)
            
            dec = 2;
            ch  = 11;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord66
        function testSynthesizerTypeIIDec11Ch5Ord66(testCase)
            
            dec = 1;
            ch  = 5;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord66
        function testSynthesizerTypeIIDec22Ch7Ord66(testCase)
            
            dec = 2;
            ch  = 7;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord66
        function testSynthesizerTypeIIDec22Ch9Ord66(testCase)
            
            dec = 2;
            ch  = 9;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord66
        function testSynthesizerTypeIIDec22Ch11Ord66(testCase)
            
            dec = 2;
            ch  = 11;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord66PeriodicExt
        function testSynthesizerTypeIIDec11Ch5Ord66PeriodicExt(testCase)
            
            dec = 1;
            ch  = 5;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border);
            imgActual = imgActual(border+1:end-border,border+1:end-border);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord66PeriodicExt
        function testSynthesizerTypeIIDec22Ch5Ord66PeriodicExt(testCase)
            
            dec = 2;
            ch  = 5;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border);
            imgActual = imgActual(border+1:end-border,border+1:end-border);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord66PeriodicExt
        function testSynthesizerTypeIIDec22Ch7Ord66PeriodicExt(testCase)
            
            dec = 2;
            ch  = 7;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border);
            imgActual = imgActual(border+1:end-border,border+1:end-border);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord66PeriodicExt
        function testSynthesizerTypeIIDec22Ch9Ord66PeriodicExt(testCase)
            
            dec = 2;
            ch  = 9;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border);
            imgActual = imgActual(border+1:end-border,border+1:end-border);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord00Level1
        function testSynthesizerTypeIIDec11Ch5Ord00Level1(testCase)
            
            dec = 1;
            ch = 5;
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
            end  ;
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch5Ord00Level1
        function testSynthesizerTypeIIDec22Ch5Ord00Level1(testCase)
            
            dec = 2;
            ch = 5;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord00Level1
        function testSynthesizerTypeIIDec22Ch7Ord00Level1(testCase)
            
            dec = 2;
            nChs= 7;
            ord = 0;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
            
        end
        
        %Dec22Ch8Ord00Level1
        function testSynthesizerTypeIIDec22Ch9Ord00Level1(testCase)
            
            dec = 2;
            ch = 9;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord00Level2
        function testSynthesizerTypeIIDec11Ch5Ord00Level2(testCase)
            
            dec = 1;
            ch = 5;
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
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord00Level2
        function testSynthesizerTypeIIDec22Ch5Ord00Level2(testCase)
            
            dec = 2;
            ch = 5;
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
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord00Level2
        function testSynthesizerTypeIIDec22Ch7Ord00Level2(testCase)
            
            dec = 2;
            ch = 7;
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
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec));
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord00Level2
        function testSynthesizerTypeIIDec22Ch9Ord00Level2(testCase)
            
            dec = 2;
            ch = 9;
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
            subCoefs{9} = rand(height/(dec^2),width/(dec^2));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec));
            subCoefs{16} = rand(height/(dec),width/(dec));
            subCoefs{17} = rand(height/(dec),width/(dec));
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord22Level1
        function testSynthesizerTypeIIDec11Ch5Ord22Level1(testCase)
            
            dec = 1;
            ch = 5;
            ord = 2;
            height = 32;
            width = 32;
            %nLevels = 1;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord22Level1
        function testSynthesizerTypeIIDec22Ch5Ord22Level1(testCase)
            
            dec = 2;
            ch = 5;
            ord = 2;
            height = 32;
            width = 32;
            %nLevels = 1;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord22Level1
        function testSynthesizerTypeIIDec22Ch7Ord22Level1(testCase)
            
            dec = 2;
            ch = 7;
            ord = 2;
            height = 32;
            width = 32;
            % nLevels = 1;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord22Level1
        function testSynthesizerTypeIIDec22Ch9Ord22Level1(testCase)
            
            dec = 2;
            ch = 9;
            ord = 2;
            height = 32;
            width = 32;
            %nLevels = 1;
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord22Level2PeriodicExt
        function testSynthesizerTypeIIDec11Ch5Ord22Level2PeriodicExt(testCase)
            
            dec = 1;
            ch = 5;
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
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord22Level2PeriodicExt
        function testSynthesizerTypeIIDec22Ch5Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 5;
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
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord22Level2PeriodicExt
        function testSynthesizerTypeIIDec22Ch7Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 7;
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
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec));
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord22Level2PeriodicExt
        function testSynthesizerTypeIIDec22Ch9Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 9;
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
            subCoefs{9} = rand(height/(dec^2),width/(dec^2));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec));
            subCoefs{16} = rand(height/(dec),width/(dec));
            subCoefs{17} = rand(height/(dec),width/(dec));
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch9Ord44Level3PeriodicExt
        function testSynthesizerTypeIIDec22Ch9Ord44Level3PeriodicExt(testCase)
            
            dec = 2;
            ch = 9;
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
            subCoefs{9} = rand(height/(dec^3),width/(dec^3));
            subCoefs{10} = rand(height/(dec^2),width/(dec^2));
            subCoefs{11} = rand(height/(dec^2),width/(dec^2));
            subCoefs{12} = rand(height/(dec^2),width/(dec^2));
            subCoefs{13} = rand(height/(dec^2),width/(dec^2));
            subCoefs{14} = rand(height/(dec^2),width/(dec^2));
            subCoefs{15} = rand(height/(dec^2),width/(dec^2));
            subCoefs{16} = rand(height/(dec^2),width/(dec^2));
            subCoefs{17} = rand(height/(dec^2),width/(dec^2));
            subCoefs{18} = rand(height/(dec),width/(dec));
            subCoefs{19} = rand(height/(dec),width/(dec));
            subCoefs{20} = rand(height/(dec),width/(dec));
            subCoefs{21} = rand(height/(dec),width/(dec));
            subCoefs{22} = rand(height/(dec),width/(dec));
            subCoefs{23} = rand(height/(dec),width/(dec));
            subCoefs{24} = rand(height/(dec),width/(dec));
            subCoefs{25} = rand(height/(dec),width/(dec));
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-8,sprintf('%g',diff));
        end
        
        %Dec11Ch7Ord88Level3PeriodicExt
        function testSynthesizerTypeIIDec11Ch9Ord88Level3PeriodicExt(testCase)
            
            dec = 1;
            ch = 9;
            ord = 8;
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
            subCoefs{9} = rand(height/(dec^3),width/(dec^3));
            subCoefs{10} = rand(height/(dec^2),width/(dec^2));
            subCoefs{11} = rand(height/(dec^2),width/(dec^2));
            subCoefs{12} = rand(height/(dec^2),width/(dec^2));
            subCoefs{13} = rand(height/(dec^2),width/(dec^2));
            subCoefs{14} = rand(height/(dec^2),width/(dec^2));
            subCoefs{15} = rand(height/(dec^2),width/(dec^2));
            subCoefs{16} = rand(height/(dec^2),width/(dec^2));
            subCoefs{17} = rand(height/(dec^2),width/(dec^2));
            subCoefs{18} = rand(height/(dec),width/(dec));
            subCoefs{19} = rand(height/(dec),width/(dec));
            subCoefs{20} = rand(height/(dec),width/(dec));
            subCoefs{21} = rand(height/(dec),width/(dec));
            subCoefs{22} = rand(height/(dec),width/(dec));
            subCoefs{23} = rand(height/(dec),width/(dec));
            subCoefs{24} = rand(height/(dec),width/(dec));
            subCoefs{25} = rand(height/(dec),width/(dec));
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch32Ord00(testCase)
            
            dec = 2;
            decch = [dec dec 3 2];
            nChs = sum(decch(3:4));
            height = 16;
            width = 16;
            subCoefs = rand(height*nChs/decch(1),width/decch(2));
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            fun = @(x) reshape(flipud(E.'*x.data(:)),decch(1),decch(2));
            imgExpctd = blockproc(subCoefs,[nChs 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch32Ord22(testCase)
            
            dec = 2;
            decch = [dec dec 3 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(1),phase).',...
                    decch(2),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(decch(1)+1:end-decch(1),decch(2)+1:end-decch(2)); % ignore border
            imgActual = imgActual(decch(1)+1:end-decch(1),decch(2)+1:end-decch(2)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch32Ord22PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(1),phase).',...
                    decch(2),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch32Ord44(testCase)
            
            dec = 2;
            decch = [dec dec 3 2];
            nChs = sum(decch(3:4));
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(1),phase).',...
                    decch(2),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*decch(1)+1:end-2*decch(1),2*decch(2)+1:end-2*decch(2)); % ignore border
            imgActual = imgActual(2*decch(1)+1:end-2*decch(1),2*decch(2)+1:end-2*decch(2)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch32Ord44PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 3 2];
            nChs = sum(decch(3:4));
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(1),phase).',...
                    decch(2),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch42Ord00(testCase)
            
            dec  = 2;
            decch = [ dec dec 4 2];
            nChs = sum(decch(3:4));
            height = 16;
            width = 16;
            subCoefs = rand(height*nChs/decch(1),width/decch(2));
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            
            % Expected values
            fun = @(x) reshape(flipud(E.'*x.data(:)),decch(1),decch(2));
            imgExpctd = blockproc(subCoefs,[nChs 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch42Ord22(testCase)
            
            dec = 2;
            decch = [dec dec 4 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(1),phase).',...
                    decch(2),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(decch(1)+1:end-decch(1),decch(2)+1:end-decch(2)); % ignore border
            imgActual = imgActual(decch(1)+1:end-decch(1),decch(2)+1:end-decch(2)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch42Ord22PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 4 2 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(1),phase).',...
                    decch(2),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch42Ord44(testCase)
            
            dec = 2;
            decch = [dec dec 4 2];
            nChs = sum(decch(3:4));
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(1),phase).',...
                    decch(2),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*decch(1)+1:end-2*decch(1),2*decch(2)+1:end-2*decch(2)); % ignore border
            imgActual = imgActual(2*decch(1)+1:end-2*decch(1),2*decch(2)+1:end-2*decch(2)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testSynthesizerTypeIIDec22Ch42Ord44PeriodicExt(testCase)
            
            dec = 2;
            decch = [dec dec 4 2];
            nChs = sum(decch(3:4));
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(1),phase).',...
                    decch(2),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch32Ord22Level1
        function testSynthesizerTypeIIDec22Ch32Ord22Level1(testCase)
            
            dec = 2;
            decch = [dec dec 3 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            %nLevels = 1;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(1),phase).',...
                    decch(2),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border1 = decch(1);
            border2 = decch(2);
            imgExpctd = imgExpctd(border1+1:end-border1,border2+1:end-border2); % ignore border
            imgActual = imgActual(border1+1:end-border1,border2+1:end-border2); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        
        %Dec22Ch32Ord22Level2PeriodicExt
        function testSynthesizerTypeIIDec32Ch5Ord22Level2PeriodicExt(testCase)
            
            decch = [2 2 3 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{2} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{3} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{4} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{5} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{6} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{7} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{8} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{9} = rand(height/(decch(1)),width/(decch(2)));
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decch(1),phase).',...
                    decch(2),phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:nChs
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decch(1),phase).',...
                        decch(2),phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec22Ch32Ord22Level1
        function testSynthesizerTypeIIDec42Ch5Ord22Level1(testCase)
            
            dec = 2;
            decch = [dec dec 4 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(1),phase).',...
                    decch(2),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border1 = decch(1);
            border2 = decch(2);
            imgExpctd = imgExpctd(border1+1:end-border1,border2+1:end-border2); % ignore border
            imgActual = imgActual(border1+1:end-border1,border2+1:end-border2); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec22Ch32Ord22Level2PeriodicExt
        function testSynthesizerTypeIIDec22Ch42Ord22Level2PeriodicExt(testCase)
            
            decch = [2 2 4 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{2} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{3} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{4} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{5} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{6} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{7} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{8} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{9} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{10} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{11} = rand(height/(decch(1)),width/(decch(2)));
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decch(1),phase).',...
                    decch(2),phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:nChs
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decch(1),phase).',...
                        decch(2),phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        function testSetLpPuFb2dDec22Ch62Ord44(testCase)
            
            dec = 2;
            ch = [ 6 2 ];
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
        function testSetLpPuFb2dDec22Ch52Ord44(testCase)
            
            dec = 2;
            ch = [ 5 2 ];
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb);
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
        function testIsCloneLpPuFb2dFalse(testCase)
            
            dec = 2;
            ch = [ 6 2 ];
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
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb,...
                'IsCloneLpPuFb2d',true);
            
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
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb,...
                'IsCloneLpPuFb2d',false);
            
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
            ch =  [ 5 3 ];
            ord = [ 4 4 ];
            height = 64;
            width  = 64;
            coefs = rand(sum(ch)/prod(dec)*height*width,1);
            scales = repmat([height/dec(1) width/dec(2)],[sum(ch) 1]);
            
            % Preparation
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.synthesizer = TypeIISynthesisSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination');

            % Clone
            cloneSynthesizer = clone(testCase.synthesizer);

            % Evaluation
            testCase.verifyEqual(cloneSynthesizer,testCase.synthesizer);
            testCase.verifyFalse(cloneSynthesizer == testCase.synthesizer);
            prpOrg = get(testCase.synthesizer,'LpPuFb2d');
            prpCln = get(cloneSynthesizer,'LpPuFb2d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            recImgExpctd = step(testCase.synthesizer,coefs,scales);
            recImgActual = step(cloneSynthesizer,coefs,scales);
            testCase.assertEqual(recImgActual,recImgExpctd);
            
        end
    end
    
end

