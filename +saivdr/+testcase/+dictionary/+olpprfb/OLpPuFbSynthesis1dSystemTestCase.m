classdef OLpPuFbSynthesis1dSystemTestCase < matlab.unittest.TestCase
    %OLpPuFbSynthesis1dSystemTESTCASE Test case for OLpPuFbSynthesis1dSystem
    %
    % SVN identifier:
    % $Id: OLpPuFbSynthesis1dSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
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
            import saivdr.dictionary.olpprfb.*
            lppufbExpctd = OvsdLpPuFb1dTypeIVm1System(...
                'OutputMode','ParameterMatrixSet');
            frmbdExpctd  = 1;
            
            % Instantiation
            testCase.synthesizer = OLpPuFbSynthesis1dSystem();
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb1d');
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
            import saivdr.dictionary.olpprfb.*
            lppufbExpctd = OvsdLpPuFb1dTypeIVm1System(...
                'NumberOfChannels',nChs,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.ChannelGroup
            testCase.synthesizer = OLpPuFbAnalysis1dSystem(...
                'NumberOfSymmetricChannels',nChs(ChannelGroup.UPPER),...
                'NumberOfAntisymmetricChannels',nChs(ChannelGroup.LOWER));
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb1d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end
        
        function testInverseBlockDctDec4(testCase)
            
            dec = 4;
            ch = dec;
            nLen = 32;
            subCoefs  = rand(dec,nLen/dec);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iCh = 1:dec
                subSeq = subCoefs(iCh,:);
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iCh) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Expected values
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec);
            E0 = step(lppufb,[],[]);
            seqExpctd = reshape(flipud(E0.'*subCoefs),[1 nLen]);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
            
        end
        
        function testStepDec2Ch22Ord0(testCase)
            
            dec = 2;
            ch = dec;
            nLen = 16;
            subCoefs = rand(dec,nLen/dec);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iCh = 1:dec
                subSeq = subCoefs(iCh,:);
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iCh) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            seqExpctd = reshape(flipud(E.'*subCoefs),[1 nLen]);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec2Ch11Ord2(testCase)
            
            dec = 2;
            ch = dec;
            ord = 2;
            nLen = 16;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(dec+1:end-dec); % ignore border
            seqActual = seqActual(dec+1:end-dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch11Ord2PeriodicExt(testCase)
            
            dec = 2;
            ch = dec;
            ord = 2;
            nLen = 16;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch22Ord2(testCase)
            
            dec = 4;
            ch = dec;
            ord = 2;
            nLen = 32;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(dec+1:end-dec); % ignore border
            seqActual = seqActual(dec+1:end-dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-8,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch22Ord2PeriodicExtVm0(testCase)
            
            dec = 4;
            ch = dec;
            ord = 2;
            nLen = 32;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch22Ord2PeriodicExtVm1(testCase)
            
            dec = 4;
            ch = dec;
            ord = 2;
            nLen = 32;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctOrd2(testCase)
            
            dec = 2;
            ch = dec;
            ord = 2;
            nLen = 16;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Expected values
            E0 = step(OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec),[],[]);
            tmp = flipud(E0.'*cell2mat(subCoefs));
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctOrd2dec4(testCase)
            
            dec = 4;
            ch = dec;
            ord = 2;
            nLen = 32;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Expected values
            E0 = step(OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec),[],[]);
            tmp = flipud(E0.'*cell2mat(subCoefs));
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch11Ord44(testCase)
            
            dec = 2;
            ch = dec;
            ord = 4;
            nLen = 16;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(2*dec+1:end-2*dec); % ignore border
            seqActual = seqActual(2*dec+1:end-2*dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch11Ord4PeriodicExt(testCase)
            
            dec = 2;
            ch = dec;
            ord = 4;
            nLen = 16;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch22Ord4(testCase)
            
            dec = 4;
            ch = dec;
            ord = 4;
            nLen = 32;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(2*dec+1:end-2*dec); % ignore border
            seqActual = seqActual(2*dec+1:end-2*dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch22Ord4PeriodicExt(testCase)
            
            dec = 4;
            ch = dec;
            ord = 4;
            nLen = 32;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testInverseBlockDctOrd4(testCase)
            
            dec = 2;
            ch = dec;
            ord = 4;
            nLen = 16;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            E0 = step(OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec),[],[]);
            tmp = flipud(E0.'*cell2mat(subCoefs));
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctOrd4Dec4(testCase)
            
            dec = 4;
            ch = dec;
            ord = 4;
            nLen = 32;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            E0 = step(...
                OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec),[],[]);
            tmp = flipud(E0.'*cell2mat(subCoefs));
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch11Ord6(testCase)
            
            dec = 2;
            ch = dec;
            ord = 6;
            nLen = 16;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(3*dec+1:end-3*dec); % ignore border
            seqActual = seqActual(3*dec+1:end-3*dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch11Ord6PeriodicExt(testCase)
            
            dec = 2;
            ch = dec;
            ord = 6;
            nLen = 16;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch22Ord6(testCase)
            
            dec = 4;
            ch = dec;
            ord = 6;
            nLen = 32;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(3*dec+1:end-3*dec); % ignore border
            seqActual = seqActual(3*dec+1:end-3*dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec4Ch22Ord6PeriodicExt(testCase)
            
            dec = 4;
            ch = dec;
            ord = 6;
            nLen = 32;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testInverseBlockDctOrd6(testCase)
            
            dec = 2;
            ch = dec;
            ord = 6;
            nLen = 16;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            E0 = step(...
                OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec),...
                [],[]);
            tmp = flipud(E0.'*cell2mat(subCoefs));
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctDec4Ch22Ord6(testCase)
            
            dec = 4;
            ch = dec;
            ord = 6;
            nLen = 32;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            E0 = step(...
                OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec),[],[]);
            tmp = flipud(E0.'*cell2mat(subCoefs));
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch11Ord4(testCase)
            
            dec = 2;
            ch = dec;
            ord = 4;
            nLen = 16;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(:,2*dec+1:end-2*dec); % ignore border
            seqActual = seqActual(:,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch22Ord2PeriodicExt(testCase)
            
            dec = 4;
            ch = dec;
            ord = 2;
            nLen = 32;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch22Ord2PeriodicExt(testCase)
            
            dec = 2;
            ch = dec;
            ord = 2;
            nLen = 16;
            subCoefs = cell(dec,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:dec
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:dec
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec1Ch4Ord0
        function testStepDec1Ch4Ord0(testCase)
            
            dec = 1;
            ch = 4;
            ord = 0;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:sum(ch)
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch4Ord0
        function testStepDec2Ch4Ord00(testCase)
            
            dec = 2;
            ch = 4;
            ord = 0;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        %Dec2Ch6Ord00
        function testStepDec2Ch6Ord0(testCase)
            
            dec = 2;
            ch = 6;
            ord = 0;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        %Dec2Ch8Ord0
        function testStepDec2Ch8Ord0(testCase)
            
            dec = 2;
            ch = 8;
            ord = 0;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        %Dec1Ch4Ord2
        function testStepDec1Ch4Ord2(testCase)
            
            dec = 1;
            ch = 4;
            ord = 2;
            nLen = 16;
            
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(dec+1:end-dec); % ignore border
            seqActual = seqActual(dec+1:end-dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch4Ord2
        function testStepDec2Ch4Ord2(testCase)
            
            dec = 2;
            ch = 4;
            ord = 2;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for ph adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch6Ord2
        function testStepDec2Ch6Ord2(testCase)
            
            dec = 2;
            ch  = 6;
            ord = 2;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch8Ord2
        function testStepDec2Ch44Ord2(testCase)
            
            dec = 2;
            ch  = 8;
            ord = 2;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec1Ch4Ord4
        function testStepDec11Ch4Ord44(testCase)
            
            dec = 1;
            ch  = 4;
            ord = 4;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch4Ord4
        function testStepDec2Ch4Ord44(testCase)
            
            dec = 2;
            ch  = 4;
            ord = 4;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch6Ord4
        function testStepDec2Ch6Ord4(testCase)
            
            dec = 2;
            ch  = 6;
            ord = 4;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch8Ord4
        function testStepDec2Ch8Ord4(testCase)
            
            dec = 2;
            ch  = 8;
            ord = 4;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec1Ch4Ord6
        function testStepDec1Ch4Ord6(testCase)
            
            dec = 1;
            ch  = 4;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch4Ord6
        function testStepDec2Ch4Ord6(testCase)
            
            dec = 2;
            ch  = 4;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch6Ord6
        function testStepDec2Ch6Ord6(testCase)
            
            dec = 2;
            ch  = 6;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    ...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch8Ord6
        function testStepDec2Ch8Ord6(testCase)
            
            dec = 2;
            ch  = 8;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec1Ch4Ord6PeriodicExt
        function testStepDec1Ch4Ord6PeriodicExt(testCase)
            
            dec = 1;
            ch  = 4;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            %border = 0;
            %seqExpctd = seqExpctd(border+1:end-border); % ignore border
            %seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch4Ord6PeriodicExt
        function testStepDec2Ch4Ord6PeriodicExt(testCase)
            
            dec = 2;
            ch  = 4;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            %border = 0;
            %seqExpctd = seqExpctd(border+1:end-border); % ignore border
            %seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch6Ord6PeriodicExt
        function testStepDec2Ch6Ord6PeriodicExt(testCase)
            
            dec = 2;
            ch  = 6;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            %border = 0;
            %seqExpctd = seqExpctd(border+1:end-border); % ignore border
            %seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch8Ord6PeriodicExt
        function testStepDec2Ch8Ord6PeriodicExt(testCase)
            
            dec = 2;
            ch  = 8;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            %border = 0;
            %seqExpctd = seqExpctd(border+1:end-border); % ignore border
            %seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec1Ch4Ord0Level1
        function testStepDec1Ch4Ord0Level1(testCase)
            
            dec = 1;
            ch = 4;
            ord = 0;
            nLen = 32;
            
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        
        %Dec2Ch4Ord00Level1
        function testStepDec2Ch4Ord00Level1(testCase)
            
            dec = 2;
            ch = 4;
            ord = 0;
            nLen = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch6Ord00Level1
        function testStepDec2Ch6Ord00Level1(testCase)
            
            dec = 2;
            ch = 6;
            ord = 0;
            nLen = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch8Ord0Level1
        function testStepDec2Ch8Ord0Level1(testCase)
            
            dec = 2;
            ch = 8;
            ord = 0;
            nLen = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord0Level2
        function testStepDec11Ch4Ord0Level2(testCase)
            
            dec = 1;
            ch = 4;
            ord = 0;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec));
            subCoefs{6} = rand(1,nLen/(dec));
            subCoefs{7} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd(:).';
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch4Ord00Level2
        function testStepDec2Ch4Ord0Level2(testCase)
            
            dec = 2;
            ch = 4;
            ord = 0;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec));
            subCoefs{6} = rand(1,nLen/(dec));
            subCoefs{7} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        
        %Dec2Ch6Ord0Level2
        function testStepDec2Ch6Ord0Level2(testCase)
            
            dec = 2;
            ch = 6;
            ord = 0;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec^2));
            subCoefs{6} = rand(1,nLen/(dec^2));
            subCoefs{7} = rand(1,nLen/(dec));
            subCoefs{8} = rand(1,nLen/(dec));
            subCoefs{9} = rand(1,nLen/(dec));
            subCoefs{10} = rand(1,nLen/(dec));
            subCoefs{11} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch8Ord00Level2
        function testStepDec2Ch8Ord0Level2(testCase)
            
            dec = 2;
            ch = 8;
            ord = 0;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec^2));
            subCoefs{6} = rand(1,nLen/(dec^2));
            subCoefs{7} = rand(1,nLen/(dec^2));
            subCoefs{8} = rand(1,nLen/(dec^2));
            subCoefs{9} = rand(1,nLen/(dec));
            subCoefs{10} = rand(1,nLen/(dec));
            subCoefs{11} = rand(1,nLen/(dec));
            subCoefs{12} = rand(1,nLen/(dec));
            subCoefs{13} = rand(1,nLen/(dec));
            subCoefs{14} = rand(1,nLen/(dec));
            subCoefs{15} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        
        %Dec1Ch4Ord2Level1
        function testStepDec1Ch4Ord2Level1(testCase)
            
            dec = 1;
            ch = 4;
            ord = 2;
            nLen = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch4Ord2Level1
        function testStepDec2Ch4Ord2Level1(testCase)
            
            dec = 2;
            ch = 4;
            ord = 2;
            nLen = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch6Ord2Level1
        function testStepDec2Ch6Ord2Level1(testCase)
            
            dec = 2;
            ch = 6;
            ord = 2;
            nLen = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch8Ord2Level1
        function testStepDec2Ch8Ord2Level1(testCase)
            
            dec = 2;
            ch = 8;
            ord = 2;
            nLen = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord2Level2PeriodicExt
        function testStepDec11Ch4Ord2Level2PeriodicExt(testCase)
            
            dec = 1;
            ch = 4;
            ord = 2;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec));
            subCoefs{6} = rand(1,nLen/(dec));
            subCoefs{7} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch4Ord2Level2PeriodicExt
        function testStepDec2Ch4Ord2Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 4;
            ord = 2;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec));
            subCoefs{6} = rand(1,nLen/(dec));
            subCoefs{7} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch6Ord2Level2PeriodicExt
        function testStepDec2Ch6Ord2Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 6;
            ord = 2;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec^2));
            subCoefs{6} = rand(1,nLen/(dec^2));
            subCoefs{7} = rand(1,nLen/(dec));
            subCoefs{8} = rand(1,nLen/(dec));
            subCoefs{9} = rand(1,nLen/(dec));
            subCoefs{10} = rand(1,nLen/(dec));
            subCoefs{11} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch8Ord4Level3PeriodicExt
        function testStepDec2Ch8Ord4Level3PeriodicExt(testCase)
            
            dec = 2;
            ch = 8;
            ord = 4;
            nLen = 64;
            nLevels = 3;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^3));
            subCoefs{2} = rand(1,nLen/(dec^3));
            subCoefs{3} = rand(1,nLen/(dec^3));
            subCoefs{4} = rand(1,nLen/(dec^3));
            subCoefs{5} = rand(1,nLen/(dec^3));
            subCoefs{6} = rand(1,nLen/(dec^3));
            subCoefs{7} = rand(1,nLen/(dec^3));
            subCoefs{8} = rand(1,nLen/(dec^3));
            subCoefs{9} = rand(1,nLen/(dec^2));
            subCoefs{10} = rand(1,nLen/(dec^2));
            subCoefs{11} = rand(1,nLen/(dec^2));
            subCoefs{12} = rand(1,nLen/(dec^2));
            subCoefs{13} = rand(1,nLen/(dec^2));
            subCoefs{14} = rand(1,nLen/(dec^2));
            subCoefs{15} = rand(1,nLen/(dec^2));
            subCoefs{16} = rand(1,nLen/(dec));
            subCoefs{17} = rand(1,nLen/(dec));
            subCoefs{18} = rand(1,nLen/(dec));
            subCoefs{19} = rand(1,nLen/(dec));
            subCoefs{20} = rand(1,nLen/(dec));
            subCoefs{21} = rand(1,nLen/(dec));
            subCoefs{22} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-8,sprintf('%g',diff));
        end
        
        function testSetLpPuFb1dDec2Ch22Ord4(testCase)
            
            dec = 2;
            decch = [ dec 4 4 ];
            ord = 4;
            nLen = 32;
            subCoefs{1} = rand(1,nLen/(dec));
            subCoefs{2} = rand(1,nLen/(dec));
            subCoefs{3} = rand(1,nLen/(dec));
            subCoefs{4} = rand(1,nLen/(dec));
            subCoefs{5} = rand(1,nLen/(dec));
            subCoefs{6} = rand(1,nLen/(dec));
            subCoefs{7} = rand(1,nLen/(dec));
            subCoefs{8} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
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
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
        function testIsCloneLpPuFb1dFalse(testCase)
            
            dec = 2;
            ch = [ 4 4 ];
            ord = 4;
            nLen = 32;
            subCoefs{1} = rand(1,nLen/(dec));
            subCoefs{2} = rand(1,nLen/(dec));
            subCoefs{3} = rand(1,nLen/(dec));
            subCoefs{4} = rand(1,nLen/(dec));
            subCoefs{5} = rand(1,nLen/(dec));
            subCoefs{6} = rand(1,nLen/(dec));
            subCoefs{7} = rand(1,nLen/(dec));
            subCoefs{8} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'IsCloneLpPuFb1d',true);
            
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
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'IsCloneLpPuFb1d',false);
            
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
            
            dec  = 2;
            ch   = [ 4 4 ];
            ord  = 4;
            nLen = 64;
            coefs = rand(1,nLen*sum(ch)/dec);
            scales = repmat(nLen/dec,[sum(ch) 1]);
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Termination');
            
            % Clone
            cloneSynthesizer = clone(testCase.synthesizer);
            
            % Evaluation
            testCase.verifyEqual(cloneSynthesizer,testCase.synthesizer);
            testCase.verifyFalse(cloneSynthesizer == testCase.synthesizer);
            prpOrg = get(testCase.synthesizer,'LpPuFb1d');
            prpCln = get(cloneSynthesizer,'LpPuFb1d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            recseqExpctd = step(testCase.synthesizer,coefs,scales);
            recseqActual = step(cloneSynthesizer,coefs,scales);
            testCase.assertEqual(recseqActual,recseqExpctd);
            
        end
        
        % Test
        function testConstructionTypeII(testCase)
            
            % Expected values
            import saivdr.dictionary.olpprfb.*
            lppufbExpctd = OvsdLpPuFb1dTypeIIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufbExpctd);
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb1d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end
        
        % Test
        function testDefaultConstruction6plus2(testCase)
            
            % Preperation
            nChs = [6 2];
            
            % Expected values
            import saivdr.dictionary.olpprfb.*
            lppufbExpctd = OvsdLpPuFb1dTypeIIVm1System(...
                'NumberOfChannels',nChs,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.ChannelGroup
            testCase.synthesizer = OLpPuFbAnalysis1dSystem(...
                'NumberOfSymmetricChannels',nChs(ChannelGroup.UPPER),...
                'NumberOfAntisymmetricChannels',nChs(ChannelGroup.LOWER));
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb1d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end
        
        
        % Test for default construction
        function testInverseBlockDctDec3(testCase)
            
            dec = 3;
            nLen = 24;
            subCoefs  = rand(dec,nLen/dec);
            coefs = zeros(1,nLen);
            scales = zeros(dec,1);
            sIdx = 1;
            for iCh = 1:dec
                subSeq = subCoefs(iCh,:);
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iCh) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Expected values
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec);
            E0 = step(lppufb,[],[]);
            tmp = flipud(E0.'*subCoefs);
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testInverseBlockDctDec5(testCase)
            
            dec = 5;
            nLen = 40;
            subCoefs = rand(dec,nLen/dec);
            coefs = zeros(1,nLen);
            scales = zeros(dec,1);
            sIdx = 1;
            for iCh = 1:dec
                subSeq = subCoefs(iCh,:);
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iCh) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Expected values
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec);
            E0 = step(lppufb,[],[]);
            tmp = flipud(E0.'*subCoefs);
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec2Ch5Ord0(testCase)
            
            dec = 2;
            nChs = 5;
            nLen = 16;
            subCoefs = rand(nChs,nLen/dec);
            coefs = zeros(1,nLen);
            scales = zeros(dec,1);
            sIdx = 1;
            for iCh = 1:nChs
                subSeq = subCoefs(iCh,:);
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iCh) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            tmp = flipud(E.'*subCoefs);
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec44Cg16Ord00(testCase)
            
            dec = 4;
            nChs = 17;
            nLen = 32;
            subCoefs = rand(nChs,nLen/dec);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iCh = 1:nChs
                subSeq = subCoefs(iCh,:);
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iCh) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', nChs,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            tmp = flipud(E.'*subCoefs);
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-8,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec2Ch5Ord2Vm0(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            nLen = 16;
            
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            import saivdr.dictionary.olpprfb.*
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(dec+1:end-dec); % ignore border
            seqActual = seqActual(dec+1:end-dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch5Ord2Vm1(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            import saivdr.dictionary.olpprfb.*
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(dec+1:end-dec); % ignore border
            seqActual = seqActual(dec+1:end-dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch5Ord2PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    ...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch17Ord2(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            nLen = 32;
            
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    ...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(dec+1:end-dec); % ignore border
            seqActual = seqActual(dec+1:end-dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch17Ord2PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            nLen = 32;
            
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        
        % Test for boundary operation
        function testInverseBlockDctDec3Ord2(testCase)
            
            dec = 3;
            ord = 2;
            nLen = 24;
            nChs = dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Expected values
            E0 = step(OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec),[],[]);
            tmp = flipud(E0.'*cell2mat(subCoefs));
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctDec55Ord2(testCase)
            
            dec = 5;
            ord = 2;
            nLen = 40;
            nChs = dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            
            % Expected values
            E0 = step(OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec),[],[]);
            tmp = flipud(E0.'*cell2mat(subCoefs));
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch5Ord44(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(2*dec+1:end-2*dec); % ignore border
            seqActual = seqActual(2*dec+1:end-2*dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch5Ord4PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch17Ord4(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            nLen = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    ...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(2*dec+1:end-2*dec); % ignore border
            seqActual = seqActual(2*dec+1:end-2*dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord44PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            nLen = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testInverseBlockDctDec3Ord4(testCase)
            
            dec = 3;
            ord = 4;
            nLen = 24;
            nChs = dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Expected values
            E0 = step(OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec),[],[]);
            tmp = flipud(E0.'*cell2mat(subCoefs));
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctDec5Ord4(testCase)
            
            dec = 5;
            ord = 4;
            nLen = 40;
            nChs = dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Expected values
            E0 = step(OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec),[],[]);
            tmp = flipud(E0.'*cell2mat(subCoefs));
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch5Ord6(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 6;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(3*dec+1:end-3*dec); % ignore border
            seqActual = seqActual(3*dec+1:end-3*dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch5Ord6PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 6;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...'
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch17Ord6(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 6;
            nLen = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(3*dec+1:end-3*dec); % ignore border
            seqActual = seqActual(3*dec+1:end-3*dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch17Ord6PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 6;
            nLen = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testInverseBlockDctDec3Ord6(testCase)
            
            dec = 3;
            ord = 6;
            nLen = 24;
            nChs = dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            
            % Expected values
            E0 = step(OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec),[],[]);
            tmp = flipud(E0.'*cell2mat(subCoefs));
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctDec5Ord6(testCase)
            
            dec = 5;
            ord = 6;
            nLen = 40;
            nChs = dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Expected values
            E0 = step(OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec),[],[]);
            tmp = flipud(E0.'*cell2mat(subCoefs));
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-8,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch5Ord2(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(:,dec+1:end-dec); % ignore border
            seqActual = seqActual(:,dec+1:end-dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch17Ord0(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 0;
            nLen = 32;
            
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(:,dec+1:end-dec); % ignore border
            seqActual = seqActual(:,dec+1:end-dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch5Ord4(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(:,2*dec+1:end-2*dec); % ignore border
            seqActual = seqActual(:,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch5Ord0PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 0;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch17Ord0PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 0;
            nLen = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch17Ord4PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            nLen = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec11Ch5Ord00
        function testStepDec1Ch5Ord0(testCase)
            
            dec = 1;
            ch = 5;
            ord = 0;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch7Ord0
        function testStepDec2Ch7Ord0(testCase)
            
            dec = 2;
            ch = 7;
            ord = 0;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        %Dec2Ch9Ord0
        function testStepDec2Ch9Ord0(testCase)
            
            dec = 2;
            ch = 9;
            ord = 0;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        
        %Dec2Ch1Ord0
        function testStepDec2Ch1Ord0(testCase)
            
            dec = 2;
            ch = 11;
            ord = 0;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        
        %Dec1Ch5Ord2
        function testStepDec1Ch5Ord2(testCase)
            
            dec = 1;
            ch = 5;
            ord = 2;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(dec+1:end-dec); % ignore border
            seqActual = seqActual(dec+1:end-dec); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch7Ord2
        function testStepDec2Ch7Ord2(testCase)
            
            dec = 2;
            ch = 7;
            ord = 2;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch9Ord2
        function testStepDec2Ch9Ord2(testCase)
            
            dec = 2;
            ch  = 9;
            ord = 2;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch11Ord2
        function testStepDec2Ch11_Ord2(testCase)
            
            dec = 2;
            ch  = 11;
            ord = 2;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    ...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        
        %Dec1Ch5Ord4
        function testStepDec1Ch5Ord4(testCase)
            
            dec = 1;
            ch  = 5;
            ord = 4;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        
        %Dec2Ch7Ord4
        function testStepDec2Ch7Ord4(testCase)
            
            dec = 2;
            ch  = 7;
            ord = 4;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch9Ord4
        function testStepDec2Ch9Ord4(testCase)
            
            dec = 2;
            ch  = 9;
            ord = 4;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch11_Ord4
        function testStepDec2Ch11_Ord4(testCase)
            
            dec = 2;
            ch  = 11;
            ord = 4;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec1Ch5Ord6
        function testStepDec1Ch5Ord6(testCase)
            
            dec = 1;
            ch  = 5;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch7Ord6
        function testStepDec2Ch7Ord6(testCase)
            
            dec = 2;
            ch  = 7;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch9Ord6
        function testStepDec2Ch9Ord6(testCase)
            
            dec = 2;
            ch  = 9;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch11Ord6
        function testStepDec2Ch11_Ord6(testCase)
            
            dec = 2;
            ch  = 11;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec1Ch5Ord6PeriodicExt
        function testStepDec1Ch5Ord6PeriodicExt(testCase)
            
            dec = 1;
            ch  = 5;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            seqExpctd = seqExpctd(border+1:end-border);
            seqActual = seqActual(border+1:end-border);
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch7Ord66PeriodicExt
        function testStepDec2Ch7Ord6PeriodicExt(testCase)
            
            dec = 2;
            ch  = 7;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            seqExpctd = seqExpctd(border+1:end-border);
            seqActual = seqActual(border+1:end-border);
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch9Ord5PeriodicExt
        function testStepDec2Ch9Ord6PeriodicExt(testCase)
            
            dec = 2;
            ch  = 9;
            ord = 6;
            nLen = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            seqExpctd = seqExpctd(border+1:end-border);
            seqActual = seqActual(border+1:end-border);
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec1Ch5Ord0Level1
        function testStepDec1Ch5Ord0Level1(testCase)
            
            dec = 1;
            ch = 5;
            ord = 0;
            nLen = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch5Ord0Level1
        function testStepDec2Ch5Ord0Level1(testCase)
            
            dec = 2;
            ch = 5;
            ord = 0;
            nLen = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch7Ord0Level1
        function testStepDec2Ch7Ord0Level1(testCase)
            
            dec = 2;
            nChs= 7;
            ord = 0;
            nLen = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
            
        end
        
        %Dec2Ch9Ord0Level1
        function testStepDec2Ch9Ord0Level1(testCase)
            
            dec = 2;
            ch = 9;
            ord = 0;
            nLen = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec1Ch5Ord0Level2
        function testStepDec1Ch5Ord0Level2(testCase)
            
            dec = 1;
            ch = 5;
            ord = 0;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec^2));
            subCoefs{6} = rand(1,nLen/(dec));
            subCoefs{7} = rand(1,nLen/(dec));
            subCoefs{8} = rand(1,nLen/(dec));
            subCoefs{9} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch5Ord0Level2
        function testStepDec2Ch5Ord0Level2(testCase)
            
            dec = 2;
            ch = 5;
            ord = 0;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec^2));
            subCoefs{6} = rand(1,nLen/(dec));
            subCoefs{7} = rand(1,nLen/(dec));
            subCoefs{8} = rand(1,nLen/(dec));
            subCoefs{9} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch7Ord0Level2
        function testStepDec2Ch7Ord0Level2(testCase)
            
            dec = 2;
            ch = 7;
            ord = 0;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec^2));
            subCoefs{6} = rand(1,nLen/(dec^2));
            subCoefs{7} = rand(1,nLen/(dec^2));
            subCoefs{8} = rand(1,nLen/(dec));
            subCoefs{9} = rand(1,nLen/(dec));
            subCoefs{10} = rand(1,nLen/(dec));
            subCoefs{11} = rand(1,nLen/(dec));
            subCoefs{12} = rand(1,nLen/(dec));
            subCoefs{13} = rand(1,nLen/(dec));
            subCoefs{14} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch9Ord0Level2
        function testStepDec2Ch9Ord0Level2(testCase)
            
            dec = 2;
            ch = 9;
            ord = 0;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec^2));
            subCoefs{6} = rand(1,nLen/(dec^2));
            subCoefs{7} = rand(1,nLen/(dec^2));
            subCoefs{8} = rand(1,nLen/(dec^2));
            subCoefs{9} = rand(1,nLen/(dec^2));
            subCoefs{10} = rand(1,nLen/(dec));
            subCoefs{11} = rand(1,nLen/(dec));
            subCoefs{12} = rand(1,nLen/(dec));
            subCoefs{13} = rand(1,nLen/(dec));
            subCoefs{14} = rand(1,nLen/(dec));
            subCoefs{15} = rand(1,nLen/(dec));
            subCoefs{16} = rand(1,nLen/(dec));
            subCoefs{17} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec1Ch4Ord2Level1
        function testStepDec1Ch5Ord2Level1(testCase)
            
            dec = 1;
            ch = 5;
            ord = 2;
            nLen = 32;
            %nLevels = 1;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch5Ord2Level1
        function testStepDec2Ch5Ord2Level1(testCase)
            
            dec = 2;
            ch = 5;
            ord = 2;
            nLen = 32;
            %nLevels = 1;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch7Ord2Level1
        function testStepDec2Ch7Ord2Level1(testCase)
            
            dec = 2;
            ch = 7;
            ord = 2;
            nLen = 32;
            % nLevels = 1;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch8Ord2Level1
        function testStepDec2Ch9Ord2Level1(testCase)
            
            dec = 2;
            ch = 9;
            ord = 2;
            nLen = 32;
            %nLevels = 1;
            subCoefs = cell(ch,1);
            coefs = zeros(1,nLen);
            scales = zeros(ch,1);
            sIdx = 1;
            for iSubband = 1:ch
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:ch
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            seqExpctd = seqExpctd(border+1:end-border); % ignore border
            seqActual = seqActual(border+1:end-border); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec1Ch5Ord2Level2PeriodicExt
        function testStepDec1Ch5Ord2Level2PeriodicExt(testCase)
            
            dec = 1;
            ch = 5;
            ord = 2;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec^2));
            subCoefs{6} = rand(1,nLen/(dec));
            subCoefs{7} = rand(1,nLen/(dec));
            subCoefs{8} = rand(1,nLen/(dec));
            subCoefs{9} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        
        %Dec2Ch5Ord2Level2PeriodicExt
        function testStepDec2Ch5Ord2Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 5;
            ord = 2;
            nLen = 32;
            
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec^2));
            subCoefs{6} = rand(1,nLen/(dec));
            subCoefs{7} = rand(1,nLen/(dec));
            subCoefs{8} = rand(1,nLen/(dec));
            subCoefs{9} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch7Ord2Level2PeriodicExt
        function testStepDec2Ch7Ord2Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 7;
            ord = 2;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec^2));
            subCoefs{6} = rand(1,nLen/(dec^2));
            subCoefs{7} = rand(1,nLen/(dec^2));
            subCoefs{8} = rand(1,nLen/(dec));
            subCoefs{9} = rand(1,nLen/(dec));
            subCoefs{10} = rand(1,nLen/(dec));
            subCoefs{11} = rand(1,nLen/(dec));
            subCoefs{12} = rand(1,nLen/(dec));
            subCoefs{13} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch9Ord2Level2PeriodicExt
        function testStepDec2Ch9Ord2Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 9;
            ord = 2;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^2));
            subCoefs{2} = rand(1,nLen/(dec^2));
            subCoefs{3} = rand(1,nLen/(dec^2));
            subCoefs{4} = rand(1,nLen/(dec^2));
            subCoefs{5} = rand(1,nLen/(dec^2));
            subCoefs{6} = rand(1,nLen/(dec^2));
            subCoefs{7} = rand(1,nLen/(dec^2));
            subCoefs{8} = rand(1,nLen/(dec^2));
            subCoefs{9} = rand(1,nLen/(dec^2));
            subCoefs{10} = rand(1,nLen/(dec));
            subCoefs{11} = rand(1,nLen/(dec));
            subCoefs{12} = rand(1,nLen/(dec));
            subCoefs{13} = rand(1,nLen/(dec));
            subCoefs{14} = rand(1,nLen/(dec));
            subCoefs{15} = rand(1,nLen/(dec));
            subCoefs{16} = rand(1,nLen/(dec));
            subCoefs{17} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch9Ord4Level3PeriodicExt
        function testStepDec2Ch9Ord4Level3PeriodicExt(testCase)
            
            dec = 2;
            ch = 9;
            ord = 4;
            nLen = 64;
            nLevels = 3;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^3));
            subCoefs{2} = rand(1,nLen/(dec^3));
            subCoefs{3} = rand(1,nLen/(dec^3));
            subCoefs{4} = rand(1,nLen/(dec^3));
            subCoefs{5} = rand(1,nLen/(dec^3));
            subCoefs{6} = rand(1,nLen/(dec^3));
            subCoefs{7} = rand(1,nLen/(dec^3));
            subCoefs{8} = rand(1,nLen/(dec^3));
            subCoefs{9} = rand(1,nLen/(dec^3));
            subCoefs{10} = rand(1,nLen/(dec^2));
            subCoefs{11} = rand(1,nLen/(dec^2));
            subCoefs{12} = rand(1,nLen/(dec^2));
            subCoefs{13} = rand(1,nLen/(dec^2));
            subCoefs{14} = rand(1,nLen/(dec^2));
            subCoefs{15} = rand(1,nLen/(dec^2));
            subCoefs{16} = rand(1,nLen/(dec^2));
            subCoefs{17} = rand(1,nLen/(dec^2));
            subCoefs{18} = rand(1,nLen/(dec));
            subCoefs{19} = rand(1,nLen/(dec));
            subCoefs{20} = rand(1,nLen/(dec));
            subCoefs{21} = rand(1,nLen/(dec));
            subCoefs{22} = rand(1,nLen/(dec));
            subCoefs{23} = rand(1,nLen/(dec));
            subCoefs{24} = rand(1,nLen/(dec));
            subCoefs{25} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-8,sprintf('%g',diff));
        end
        
        %Dec1Ch9Ord8Level3PeriodicExt
        function testStepDec1Ch9Ord8Level3PeriodicExt(testCase)
            
            dec = 1;
            ch = 9;
            ord = 8;
            nLen = 64;
            nLevels = 3;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(1,nLen/(dec^3));
            subCoefs{2} = rand(1,nLen/(dec^3));
            subCoefs{3} = rand(1,nLen/(dec^3));
            subCoefs{4} = rand(1,nLen/(dec^3));
            subCoefs{5} = rand(1,nLen/(dec^3));
            subCoefs{6} = rand(1,nLen/(dec^3));
            subCoefs{7} = rand(1,nLen/(dec^3));
            subCoefs{8} = rand(1,nLen/(dec^3));
            subCoefs{9} = rand(1,nLen/(dec^3));
            subCoefs{10} = rand(1,nLen/(dec^2));
            subCoefs{11} = rand(1,nLen/(dec^2));
            subCoefs{12} = rand(1,nLen/(dec^2));
            subCoefs{13} = rand(1,nLen/(dec^2));
            subCoefs{14} = rand(1,nLen/(dec^2));
            subCoefs{15} = rand(1,nLen/(dec^2));
            subCoefs{16} = rand(1,nLen/(dec^2));
            subCoefs{17} = rand(1,nLen/(dec^2));
            subCoefs{18} = rand(1,nLen/(dec));
            subCoefs{19} = rand(1,nLen/(dec));
            subCoefs{20} = rand(1,nLen/(dec));
            subCoefs{21} = rand(1,nLen/(dec));
            subCoefs{22} = rand(1,nLen/(dec));
            subCoefs{23} = rand(1,nLen/(dec));
            subCoefs{24} = rand(1,nLen/(dec));
            subCoefs{25} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-8,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch32Ord0(testCase)
            
            dec = 2;
            decch = [dec 3 2];
            nChs = sum(decch(2:3));
            nLen = 16;
            
            subCoefs = rand(nChs,nLen/dec);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iCh = 1:nChs
                subSeq = subCoefs(iCh,:);
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iCh) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            tmp = flipud(E.'*subCoefs);
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec2Ch32Ord2(testCase)
            
            dec = 2;
            decch = [dec 3 2];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(decch(1)+1:end-decch(1)); % ignore border
            seqActual = seqActual(decch(1)+1:end-decch(1)); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch32Ord2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec 3 2 ];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch32Ord4(testCase)
            
            dec = 2;
            decch = [ dec 3 2];
            nChs = sum(decch(2:3));
            ord = 4;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(2*decch(1)+1:end-2*decch(1)); % ignore border
            seqActual = seqActual(2*decch(1)+1:end-2*decch(1)); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch32Ord4PeriodicExt(testCase)
            
            dec = 2;
            decch = [  dec 3 2];
            nChs = sum(decch(2:3));
            ord = 4;
            nLen = 16;
            
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch42Ord0(testCase)
            
            dec  = 2;
            decch = [ dec 4 2];
            nChs = sum(decch(2:3));
            nLen = 16;
            subCoefs = rand(nChs,nLen/dec);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iCh = 1:nChs
                subSeq = subCoefs(iCh,:);
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iCh) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3));
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            tmp = flipud(E.'*subCoefs);
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec2Ch42Ord2(testCase)
            
            dec = 2;
            decch = [dec 4 2];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(decch(1)+1:end-decch(1)); % ignore border
            seqActual = seqActual(decch(1)+1:end-decch(1)); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch42Ord2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec 4 2 ];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch42Ord4(testCase)
            
            dec = 2;
            decch = [dec 4 2];
            nChs = sum(decch(2:3));
            ord = 4;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(2*decch(1)+1:end-2*decch(1)); % ignore border
            seqActual = seqActual(2*decch(1)+1:end-2*decch(1)); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch42Ord4PeriodicExt(testCase)
            
            dec = 2;
            decch = [dec 4 2];
            nChs = sum(decch(2:3));
            ord = 4;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch32Ord2Level1
        function testStepDec2Ch32Ord2Level1(testCase)
            
            dec = 2;
            decch = [dec 3 2];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            %nLevels = 1;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border1 = decch(1);
            seqExpctd = seqExpctd(border1+1:end-border1); % ignore border
            seqActual = seqActual(border1+1:end-border1); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        %Dec2Ch32Ord2Level2PeriodicExt
        function testStepDec2Ch32Ord2Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [dec 3 2];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(1,nLen/(decch(1)^2));
            subCoefs{2} = rand(1,nLen/(decch(1)^2));
            subCoefs{3} = rand(1,nLen/(decch(1)^2));
            subCoefs{4} = rand(1,nLen/(decch(1)^2));
            subCoefs{5} = rand(1,nLen/(decch(1)^2));
            subCoefs{6} = rand(1,nLen/decch(1));
            subCoefs{7} = rand(1,nLen/decch(1));
            subCoefs{8} = rand(1,nLen/decch(1));
            subCoefs{9} = rand(1,nLen/decch(1));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},dec,phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:nChs
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch32Ord2Level1
        function testStepDec42Ch5Ord2Level1(testCase)
            
            dec = 2;
            decch = [dec 4 2];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border1 = decch(1);
            seqExpctd = seqExpctd(border1+1:end-border1); % ignore border
            seqActual = seqActual(border1+1:end-border1); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch32Ord2Level2PeriodicExt
        function testStepDec2Ch42Ord2Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [dec 4 2];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(1,nLen/(decch(1)^2));
            subCoefs{2} = rand(1,nLen/(decch(1)^2));
            subCoefs{3} = rand(1,nLen/(decch(1)^2));
            subCoefs{4} = rand(1,nLen/(decch(1)^2));
            subCoefs{5} = rand(1,nLen/(decch(1)^2));
            subCoefs{6} = rand(1,nLen/(decch(1)^2));
            subCoefs{7} = rand(1,nLen/decch(1));
            subCoefs{8} = rand(1,nLen/decch(1));
            subCoefs{9} = rand(1,nLen/decch(1));
            subCoefs{10} = rand(1,nLen/decch(1));
            subCoefs{11} = rand(1,nLen/decch(1));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                period = nLen/(dec^(nLevels-iLevel));
                seqExpctd = circshift(cconv(...
                    upsample(subsubCoefs{1},decch(1),phs).',...
                    step(lppufb,[],[],1),period),offset).';
                for iSubSub = 2:nChs
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    subSeq = circshift(cconv(...
                        upsample(subCoefs{iSubband},dec,phs).',...
                        step(lppufb,[],[],iSubSub),period),offset).';
                    seqExpctd = seqExpctd + subSeq;
                end
                subsubCoefs{1}=seqExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        function testSetLpPuFb1dDec2Ch62Ord4(testCase)
            
            dec = 2;
            ch = [ 6 2 ];
            ord = 4;
            nLen = 32;
            subCoefs{1} = rand(1,nLen/(dec));
            subCoefs{2} = rand(1,nLen/(dec));
            subCoefs{3} = rand(1,nLen/(dec));
            subCoefs{4} = rand(1,nLen/(dec));
            subCoefs{5} = rand(1,nLen/(dec));
            subCoefs{6} = rand(1,nLen/(dec));
            subCoefs{7} = rand(1,nLen/(dec));
            subCoefs{8} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
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
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
        function testSetLpPuFb1dDec2Ch52Ord4(testCase)
            
            dec = 2;
            ch = [ 5 2 ];
            ord = 4;
            nLen = 32;
            subCoefs{1} = rand(1,nLen/(dec));
            subCoefs{2} = rand(1,nLen/(dec));
            subCoefs{3} = rand(1,nLen/(dec));
            subCoefs{4} = rand(1,nLen/(dec));
            subCoefs{5} = rand(1,nLen/(dec));
            subCoefs{6} = rand(1,nLen/(dec));
            subCoefs{7} = rand(1,nLen/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband) = length(subCoefs{iSubband});
                eIdx = sIdx + scales(iSubband)-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb);
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
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
        % Test Clone
        function testCloneTypeII(testCase)
            
            dec = 2;
            ch =  [ 5 3 ];
            ord = 4;
            nLen = 64;
            coefs = rand(sum(ch),nLen/dec);
            scales = repmat(nLen/dec,[sum(ch) 1]);
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Termination');
            
            % Clone
            cloneSynthesizer = clone(testCase.synthesizer);
            
            % Evaluation
            testCase.verifyEqual(cloneSynthesizer,testCase.synthesizer);
            testCase.verifyFalse(cloneSynthesizer == testCase.synthesizer);
            prpOrg = get(testCase.synthesizer,'LpPuFb1d');
            prpCln = get(cloneSynthesizer,'LpPuFb1d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            recseqExpctd = step(testCase.synthesizer,coefs,scales);
            recseqActual = step(cloneSynthesizer,coefs,scales);
            testCase.assertEqual(recseqActual,recseqExpctd);
            
        end
        
        % Test
        function testStepDec2Ch23Ord0(testCase)
            
            dec = 2;
            decch = [dec 2 3];
            nChs = sum(decch(2:3));
            nLen = 16;
            subCoefs = rand(nChs,nLen/dec);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iCh = 1:nChs
                subSeq = subCoefs(iCh,:);
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iCh) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3));
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            tmp = flipud(E.'*subCoefs);
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec2Ch23Ord2(testCase)
            
            dec = 2;
            decch = [dec 2 3];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(decch(1)+1:end-decch(1)); % ignore border
            seqActual = seqActual(decch(1)+1:end-decch(1)); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch23Ord2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec 2 3 ];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch23Ord4(testCase)
            
            dec = 2;
            decch = [dec 2 3];
            nChs = sum(decch(2:3));
            ord = 4;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(2*decch(1)+1:end-2*decch(1)); % ignore border
            seqActual = seqActual(2*decch(1)+1:end-2*decch(1)); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch23Ord4PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec 2 3];
            nChs = sum(decch(2:3));
            ord = 4;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch24Ord0(testCase)
            
            dec  = 2;
            decch = [ dec 2 4 ];
            nChs = sum(decch(2:3));
            nLen = 16;
            subCoefs = rand(nChs,nLen/decch(1));
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iCh = 1:nChs
                subSeq = subCoefs(iCh,:);
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iCh) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3));
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            tmp = flipud(E.'*subCoefs);
            seqExpctd = tmp(:).';
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec2Ch24Ord2(testCase)
            
            dec = 2;
            decch = [dec 2 4];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(decch(1)+1:end-decch(1)); % ignore border
            seqActual = seqActual(decch(1)+1:end-decch(1)); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch24Ord2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec 2 4 ];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch24Ord4(testCase)
            
            dec = 2;
            decch = [dec 2 4];
            nChs = sum(decch(2:3));
            ord = 4;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            seqExpctd = seqExpctd(2*decch(1)+1:end-2*decch(1)); % ignore border
            seqActual = seqActual(2*decch(1)+1:end-2*decch(1)); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec2Ch24Ord4PeriodicExt(testCase)
            
            dec = 2;
            decch = [dec 2 4];
            nChs = sum(decch(2:3));
            ord = 4;
            nLen = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:3),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-9,sprintf('%g',diff));
        end
        
        %Dec2Ch32Ord2Level1
        function testStepDec2Ch23Ord2Level1(testCase)
            
            dec = 2;
            decch = [ dec 2 3];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            %nLevels = 1;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,nLen);
            scales = zeros(nChs,1);
            sIdx = 1;
            for iSubband = 1:nChs
                subSeq = rand(1,nLen/dec);
                subCoefs{iSubband} = subSeq;
                eIdx = sIdx + length(subSeq) - 1;
                coefs(sIdx:eIdx) = subSeq(:).';
                scales(iSubband) = length(subSeq);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','SynthesisFilterAt');
            seqExpctd = zeros(1,nLen);
            offset = -dec*ord/2;
            phs = 0; % for phs adjustment required experimentaly
            for iSubband = 1:nChs
                subSeq = circshift(cconv(...
                    upsample(subCoefs{iSubband},dec,phs).',...
                    step(lppufb,[],[],iSubband),nLen),offset).';
                seqExpctd = seqExpctd + subSeq;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = OLpPuFbSynthesis1dSystem(....
                'LpPuFb1d',lppufb);
            
            % Actual values
            seqActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(seqActual,size(seqExpctd),...
                'Actual image size is different from the expected one.');
            border1 = decch(1);
            seqExpctd = seqExpctd(border1+1:end-border1); % ignore border
            seqActual = seqActual(border1+1:end-border1); % ignore border
            diff = max(abs(seqExpctd(:) - seqActual(:))./abs(seqExpctd(:)));
            testCase.verifyEqual(seqActual,seqExpctd,'RelTol',1e-10,sprintf('%g',diff));
        end
        
    end
    
end
