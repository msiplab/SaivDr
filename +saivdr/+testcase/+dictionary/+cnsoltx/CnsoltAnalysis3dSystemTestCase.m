classdef CnsoltAnalysis3dSystemTestCase < matlab.unittest.TestCase
    %NsoltAnalysis3dSystemTESTCASE Test case for NsoltAnalysis3dSystem
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2014-2016, Shogo MURAMATSU
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
        function testDefaultConstructionTypeI(testCase)
            
            % Expected values
            import saivdr.dictionary.cnsoltx.*
            lppufbExpctd = CplxOvsdLpPuFb3dTypeIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            testCase.analyzer = CnsoltAnalysis3dSystem();
            
            % Actual value
            lppufbActual = get(testCase.analyzer,'LpPuFb3d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end

        % Test
        function testDefaultConstruction4plus4(testCase)
            
            % Preperation
            nChs = 6;
            
            % Expected values
            import saivdr.dictionary.cnsoltx.*
            lppufbExpctd = CplxOvsdLpPuFb3dTypeIVm1System(...
                'NumberOfChannels',nChs,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.ChannelGroup
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'NumberOfChannels',nChs);
            
            % Actual value
            lppufbActual = get(testCase.analyzer,'LpPuFb3d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end        

        % Test
        function testStepDec111Ch22Ord000Level1Vm0(testCase)
            
            dec = 1;
            ch = 4;
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(dec*dec*dec);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                            imfilter(srcImg,atom,'conv','circ'),...
                            1);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./[dec dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end

        % Test
        function testStepDec111Ch22Ord000Level1Vm1(testCase)
            
            dec = 1;
            ch = 4;
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(dec*dec);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./[dec dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end

        % Test
        function testStepDec111Ch22Ord000Level2PeriodicExtVm0(testCase)
            
            dec = 1;
            ch = 8;
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);            
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,...
                    'conv','circ'),dec);
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
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));  
            
        end
            
        % Test
        function testStepDec111Ch22Ord000Level2PeriodicExtVm1(testCase)
            
            dec = 1;
            ch = 4;
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);            
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};            
            coefs{7} = coefsExpctdLv1{4};
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
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));  
            
        end

        % Test
        function testStepDec222Ch44Ord000Level1PeriodicExt(testCase)
            
            dec = 2;
            ch = 8;
            
            ord = 0;
            height = 32;
            width  = 32;
            depth  = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
           % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);            
           release(lppufb)
           set(lppufb,'OutputMode','AnalysisFilterAt');
           nSubCoefs = numel(srcImg)/(dec*dec*dec);
           coefsExpctd = zeros(1,ch*nSubCoefs);
           for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
           end
            scalesExpctd = repmat(size(srcImg)./[dec dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-7,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec222Ch44Ord222Level1PeriodicExt(testCase)
            
            dec = 2;
            ch = 8;
            
            ord = 2;
            height = 16;
            width  = 32;
            depth  = 64;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
           % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);            
           release(lppufb)
           set(lppufb,'OutputMode','AnalysisFilterAt');
           nSubCoefs = numel(srcImg)/(dec*dec*dec);
           coefsExpctd = zeros(1,ch*nSubCoefs);
           for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
           end
            scalesExpctd = repmat(size(srcImg)./[dec dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-7,sprintf('%g',diff));
            
        end        
        
        % Test
        function testStepDec222Ch66Ord222Level1PeriodicExt(testCase)
            
            dec = 2;
            ch = 12;
            
            ord = 2;
            height = 32;
            width  = 32;
            depth  = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
           % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);            
           release(lppufb)
           set(lppufb,'OutputMode','AnalysisFilterAt');
           nSubCoefs = numel(srcImg)/(dec*dec*dec);
           coefsExpctd = zeros(1,ch*nSubCoefs);
           for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
           end
            scalesExpctd = repmat(size(srcImg)./[dec dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-7,sprintf('%g',diff));
            
        end                
        
        % Test
        function testStepDec222Ch44Ord222Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 8;
            ord = 2;
            height = 32;
            width  = 32;
            depth  = 32;
            srcImg = rand(height,width,depth);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);             
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),...
                    dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
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
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));  

        end
        
        % Test
        function testStepDec222Ch66Ord222Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 12;
            ord = 2;
            height = 32;
            width  = 32;
            depth  = 32;
            srcImg = rand(height,width,depth);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);             
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),...
                    dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
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
            coefs{10} = coefsExpctdLv2{10};                        
            coefs{11} = coefsExpctdLv2{11};                                    
            coefs{12} = coefsExpctdLv2{12};                                                
            coefs{13} = coefsExpctdLv1{2};
            coefs{14} = coefsExpctdLv1{3};
            coefs{15} = coefsExpctdLv1{4};
            coefs{16} = coefsExpctdLv1{5};
            coefs{17} = coefsExpctdLv1{6};
            coefs{18} = coefsExpctdLv1{7}; 
            coefs{19} = coefsExpctdLv1{8};
            coefs{20} = coefsExpctdLv1{9};
            coefs{21} = coefsExpctdLv1{10}; 
            coefs{22} = coefsExpctdLv1{11};                                    
            coefs{23} = coefsExpctdLv1{12};                                                
            
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
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));  

        end        
        
        % Test
        function testStepDec222Ch44Ord222Level3PeriodicExt(testCase)
            
            dec = 2;
            ch =  8;
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);                         
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);                
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
            end      
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);                                
                coefsExpctdLv3{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv2{1},atom,'conv','circ'),dec);
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
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end        
        
        % Test
        function testStepDec222Ch66Ord222Level3PeriodicExt(testCase)
            
            dec = 2;
            ch =  12;
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);                         
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);                
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
            end      
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);                                
                coefsExpctdLv3{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv2{1},atom,'conv','circ'),dec);
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
            coefs{11} = coefsExpctdLv3{11};
            coefs{12} = coefsExpctdLv3{12};
            coefs{13} = coefsExpctdLv2{2};
            coefs{14} = coefsExpctdLv2{3};
            coefs{15} = coefsExpctdLv2{4};
            coefs{16} = coefsExpctdLv2{5};
            coefs{17} = coefsExpctdLv2{6};
            coefs{18} = coefsExpctdLv2{7};
            coefs{19} = coefsExpctdLv2{8};            
            coefs{20} = coefsExpctdLv2{9};
            coefs{21} = coefsExpctdLv2{10};            
            coefs{22} = coefsExpctdLv2{11};
            coefs{23} = coefsExpctdLv2{12};                        
            coefs{24} = coefsExpctdLv1{2};
            coefs{25} = coefsExpctdLv1{3};
            coefs{26} = coefsExpctdLv1{4};            
            coefs{27} = coefsExpctdLv1{5};
            coefs{28} = coefsExpctdLv1{6};
            coefs{29} = coefsExpctdLv1{7};
            coefs{30} = coefsExpctdLv1{8};
            coefs{31} = coefsExpctdLv1{9};
            coefs{32} = coefsExpctdLv1{10};
            coefs{33} = coefsExpctdLv1{11};
            coefs{34} = coefsExpctdLv1{12};            
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
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end              
        
        % Test
        function testStepDec222Ch66Ord444Level3PeriodicExt(testCase)
            
            dec = 2;
            ch =  12;
            ord = 4;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);                         
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);                
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
            end      
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);                                
                coefsExpctdLv3{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv2{1},atom,'conv','circ'),dec);
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
            coefs{11} = coefsExpctdLv3{11};
            coefs{12} = coefsExpctdLv3{12};
            coefs{13} = coefsExpctdLv2{2};
            coefs{14} = coefsExpctdLv2{3};
            coefs{15} = coefsExpctdLv2{4};
            coefs{16} = coefsExpctdLv2{5};
            coefs{17} = coefsExpctdLv2{6};
            coefs{18} = coefsExpctdLv2{7};
            coefs{19} = coefsExpctdLv2{8};            
            coefs{20} = coefsExpctdLv2{9};
            coefs{21} = coefsExpctdLv2{10};            
            coefs{22} = coefsExpctdLv2{11};
            coefs{23} = coefsExpctdLv2{12};                        
            coefs{24} = coefsExpctdLv1{2};
            coefs{25} = coefsExpctdLv1{3};
            coefs{26} = coefsExpctdLv1{4};            
            coefs{27} = coefsExpctdLv1{5};
            coefs{28} = coefsExpctdLv1{6};
            coefs{29} = coefsExpctdLv1{7};
            coefs{30} = coefsExpctdLv1{8};
            coefs{31} = coefsExpctdLv1{9};
            coefs{32} = coefsExpctdLv1{10};
            coefs{33} = coefsExpctdLv1{11};
            coefs{34} = coefsExpctdLv1{12};            
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
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end              
                
        % Test
        function testClone(testCase)
            
            dec = [ 2 2 2 ];
            ch =  8;
            ord = [ 4 4 4 ];
            height = 64;
            width  = 64;
            depth  = 64;
            nLevels = 1;
            srcImg = rand(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'BoundaryOperation','Termination');
            %s = matlab.System.saveObject(testCase.analyzer);
            
            % Clone
            cloneAnalyzer = clone(testCase.analyzer);
            
            % Evaluation
            testCase.verifyEqual(cloneAnalyzer,testCase.analyzer);
            testCase.verifyFalse(cloneAnalyzer == testCase.analyzer);
            prpOrg = get(testCase.analyzer,'LpPuFb3d');
            prpCln = get(cloneAnalyzer,'LpPuFb3d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            [coefExpctd,scaleExpctd] = step(testCase.analyzer,srcImg,nLevels);
            [coefActual,scaleActual] = step(cloneAnalyzer,srcImg,nLevels);
            testCase.assertEqual(coefActual,coefExpctd);
            testCase.assertEqual(scaleActual,scaleExpctd);
            
        end     

       % Test
        function testDefaultConstructionTypeII(testCase)
            
            % Expected values
            import saivdr.dictionary.cnsoltx.*
            lppufbExpctd = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufbExpctd);
            
            % Actual value
            lppufbActual = get(testCase.analyzer,'LpPuFb3d');
            
            % Evaluation
            testCase.verifyEqual(lppufbActual,lppufbExpctd);
        end
        
        % Test
        function testStepDec222Ch54Ord222Level1PeriodicExt(testCase)
            
            dec = 2;
            ch = 9;
            
            ord = 2;
            height = 32;
            width  = 32;
            depth  = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
           % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);            
           release(lppufb)
           set(lppufb,'OutputMode','AnalysisFilterAt');
           nSubCoefs = numel(srcImg)/(dec*dec*dec);
           coefsExpctd = zeros(1,ch*nSubCoefs);
           for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
           end
            scalesExpctd = repmat(size(srcImg)./[dec dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-7,sprintf('%g',diff));
            
        end                
                        
        % Test
        function testStepDec111Ch32Ord000Level1PeriodicExtVm0(testCase)
            
            dec = 1;
            decch = [ dec dec dec 5 ];
            ch = sum(decch(4));
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4),...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);            
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(dec*dec*dec);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:3),ch,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec111Ch32Ord000Level1PeriodicExtVm1(testCase)
            
            dec = 1;
            decch = [ dec dec dec 5 ];
            ch = sum(decch(4));
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nvm = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4),...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',nvm);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);            
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(dec*dec*dec);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:3),ch,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec111Ch32Ord000Level2PeriodicExtVm0(testCase)
            
            dec = 1;
            decch = [ dec dec dec 5 ];
            ch = sum(decch(4));
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels', decch(4),...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);                        
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
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
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testStepDec111Ch32Ord000Level2PeriodicExtVm1(testCase)
            
            dec = 1;
            decch = [ dec dec dec 5 ];
            ch = sum(decch(4));
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels', decch(4),...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);                                    
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
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
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        %{
        % Test
        function testStepDec444Ch98Ord22Level1(testCase)
            
            dec = 4;
            decch = [ dec dec 9 8 ];
            ch = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2),1).',decch(1),1);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),ch,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        %}
        
        % Test
        function testStepDec222Ch54Ord000Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec dec 9];
            ch = sum(decch(4));
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);                                                
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
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
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec111Ch32Ord222Level1PeriodicExt(testCase)
            
            dec = 1;
            decch = [ dec dec dec 5 ];
            ch = sum(decch(4));
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4),...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);            
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(decch(1:3));
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:3),ch,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

        % Test
        function testStepDec111Ch32Ord222Level2PeriodicExt(testCase)
            
            dec = 1;
            decch = [ dec dec dec 5 ];
            ch = sum(decch(4));
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);                        
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
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
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testStepDec222Ch54Ord222Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec dec 9 ];
            ch = sum(decch(4));
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);                        
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
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
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
            
        end

        % Test
        function testStepDec222Ch54Ord222Level3PeriodicExt(testCase)
            
            dec = 2;
            ch = 9;
            decch = [ dec dec dec ch ];
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);                                    
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
            end
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv3{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv2{1},atom,'conv','circ'),dec);
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
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-6,...
                sprintf('%g',diff));
            
        end

        % Test
        function testStepDec222Ch54Ord444Level3PeriodicExt(testCase)
            
            dec = 2;
            ch = 9;
            decch = [ dec dec dec ch ];
            ord = 4;
            height = 64;
            width = 64;
            depth = 64;
            srcImg = rand(height,width,depth);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv1{iSubband} = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv2{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv1{1},atom,'conv','circ'),dec);
            end
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                coefsExpctdLv3{iSubband} = downsample3_(...
                    imfilter(coefsExpctdLv2{1},atom,'conv','circ'),dec);
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
            scalesExpctd = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-6,...
                sprintf('%g',diff));
            
        end
% 
%         % Test
%         function testIsCloneFalseTypeII(testCase)
%             
%             dec = [ 2 2 2 ];
%             ch =  [ 6 4 ];
%             ord = [ 4 4 4 ];
%             height = 64;
%             width = 64;
%             depth = 64;
%             nLevels = 1;
%             srcImg = rand(height,width,depth);
%             
%             % Preparation
%             import saivdr.dictionary.cnsoltx.*
%             lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
%                 'DecimationFactor',dec,...
%                 'NumberOfChannels',ch,...
%                 'PolyPhaseOrder',ord,...
%                 'OutputMode','ParameterMatrixSet');
%             
%             % Instantiation of target class
%             testCase.analyzer = CnsoltAnalysis3dSystem(...
%                 'LpPuFb3d',lppufb,...
%                 'BoundaryOperation','Termination',...
%                 'IsCloneLpPuFb3d',true);
%             
%             % Pre
%             coefsPre1 = step(testCase.analyzer,srcImg,nLevels);
%             
%             % Pst
%             angs = randn(size(get(lppufb,'Angles')));
%             set(lppufb,'Angles',angs);
%             coefsPst1 = step(testCase.analyzer,srcImg,nLevels);
%             
%             % Evaluation
%             diff = norm(coefsPst1(:)-coefsPre1(:));
%             testCase.verifyEqual(diff,0,'AbsTol',1e-15,sprintf('%g',diff));
%             
%             % Instantiation of target class
%             testCase.analyzer = CnsoltAnalysis3dSystem(...
%                 'LpPuFb3d',lppufb,...
%                 'BoundaryOperation','Termination',...
%                 'IsCloneLpPuFb3d',false);
%             
%             % Pre
%             coefsPre1 = step(testCase.analyzer,srcImg,nLevels);
%             
%             % Pst
%             angs = randn(size(get(lppufb,'Angles')));
%             set(lppufb,'Angles',angs);
%             coefsPst1 = step(testCase.analyzer,srcImg,nLevels);
%             
%             % Evaluation
%             import matlab.unittest.constraints.IsGreaterThan
%             diff = norm(coefsPst1(:)-coefsPre1(:));
%             testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));            
%            
%         end
% 
%         % Test
%         function testCloneTypeII(testCase)
%             
%             dec = [ 2 2 2 ];
%             ch =  [ 6 4 ];
%             ord = [ 4 4 4 ];
%             height = 64;
%             width  = 64;
%             depth  = 64;
%             nLevels = 1;
%             srcImg = rand(height,width,depth);
%             
%             % Preparation
%             import saivdr.dictionary.cnsoltx.*
%             lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
%                 'DecimationFactor',dec,...
%                 'NumberOfChannels',ch,...
%                 'PolyPhaseOrder',ord,...
%                 'OutputMode','ParameterMatrixSet');
%             
%             % Instantiation of target class
%             testCase.analyzer = CnsoltAnalysis3dSystem(...
%                 'LpPuFb3d',lppufb,...
%                 'BoundaryOperation','Termination');
%             %s = matlab.System.saveObject(testCase.analyzer);
% 
%             % Clone
%             cloneAnalyzer = clone(testCase.analyzer);
%             
%             % Evaluation
%             testCase.verifyEqual(cloneAnalyzer,testCase.analyzer);
%             testCase.verifyFalse(cloneAnalyzer == testCase.analyzer);
%             prpOrg = get(testCase.analyzer,'LpPuFb3d');
%             prpCln = get(cloneAnalyzer,'LpPuFb3d');
%             testCase.verifyEqual(prpCln,prpOrg);
%             testCase.verifyFalse(prpCln == prpOrg);
%             %            
%             [coefExpctd,scaleExpctd] = step(testCase.analyzer,srcImg,nLevels);
%             [coefActual,scaleActual] = step(cloneAnalyzer,srcImg,nLevels);
%             testCase.assertEqual(coefActual,coefExpctd);
%             testCase.assertEqual(scaleActual,scaleExpctd);
%             
%         end
        
        %{
        % Test
        function testStepDec444Ch3232Ord000(testCase)
            
            dec = 4;
            ch = [ 32 32 ];
            ch = sum(ch);
            ord = 0;
            height = 16;
            width  = 16;
            depth  = 16;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',[dec dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
           % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d),1),d),1),d),1);            
           release(lppufb)
           set(lppufb,'OutputMode','AnalysisFilterAt');
           nSubCoefs = numel(srcImg)/(dec*dec*dec);
           coefsExpctd = zeros(1,ch*nSubCoefs);
           for iSubband = 1:ch
                atom = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
           end
            scalesExpctd = repmat(size(srcImg)./[dec dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'NumberOfAntisymmetricChannels',ch(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-7,sprintf('%g',diff));
            
        end
        %}
        
        %{
        % Test
        function testStepDec444Ch98Ord22Level1(testCase)
            
            dec = 4;
            decch = [ dec dec 9 8 ];
            ch = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2),1).',decch(1),1);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),ch,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        %}
                
        % Test
        function testStepDec112Ch22Ord000Level1Vm0(testCase)
            
            nDecs = [ 1 1 2 ];
            ch = 4;
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                            imfilter(srcImg,atom,'conv','circ'),...
                            nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end
        
        % Test
        function testStepDec121Ch22Ord000Level1Vm0(testCase)
            
            nDecs = [ 1 2 1 ];
            ch = 4;
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                            imfilter(srcImg,atom,'conv','circ'),...
                            nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end
        
        % Test
        function testStepDec211Ch22Ord000Level1Vm0(testCase)
            
            nDecs = [ 2 1 1 ];
            ch = 4;
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                            imfilter(srcImg,atom,'conv','circ'),...
                            nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end     
        
        % Test
        function testStepDec112Ch22Ord222Level1Vm0(testCase)
            
            nDecs = [ 1 1 2 ];
            ch = 4;
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                            imfilter(srcImg,atom,'conv','circ'),...
                            nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end
        
        % Test
        function testStepDec121Ch22Ord222Level1Vm0(testCase)
            
            nDecs = [ 1 2 1 ];
            ch = 4;
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                            imfilter(srcImg,atom,'conv','circ'),...
                            nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end
        
        % Test
        function testStepDec211Ch22Ord222Level1Vm0(testCase)
            
            nDecs = [ 2 1 1 ];
            ch = 4;
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                            imfilter(srcImg,atom,'conv','circ'),...
                            nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end
        
        % Test
        function testStepDec112Ch32Ord000Level1Vm0(testCase)
            
            nDecs = [ 1 1 2 ];
            ch = 5;
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),...
                    nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,ch,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec121Ch32Ord000Level1Vm0(testCase)
            
            nDecs = [ 1 2 1 ];
            ch = 5;
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),...
                    nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,ch,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec211Ch32Ord000Level1Vm0(testCase)
            
            nDecs = [ 2 1 1 ];
            ch = 5;
            ord = 0;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),...
                    nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,ch,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec112Ch32Ord222Level1Vm0(testCase)
            
            nDecs = [ 1 1 2 ];
            ch = 5;
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),...
                    nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,ch,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec121Ch32Ord222Level1Vm0(testCase)
            
            nDecs = [ 1 2 1 ];
            ch = 5;
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),...
                    nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,ch,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec211Ch32Ord222Level1Vm0(testCase)
            
            nDecs = [ 2 1 1 ];
            ch = 5;
            ord = 2;
            height = 32;
            width = 32;
            depth = 32;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            downsample3_ = @(x,d) ...
                shiftdim(downsample(...
                shiftdim(downsample(...
                shiftdim(downsample(x,d(1)),1),d(2)),1),d(3)),1);
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/prod(nDecs);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                atom  = step(lppufb,[],[],iSubband);
                subCoef = downsample3_(...
                    imfilter(srcImg,atom,'conv','circ'),...
                    nDecs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./nDecs,ch,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = CnsoltAnalysis3dSystem(...
                'LpPuFb3d',lppufb,...
                'NumberOfChannels',ch,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg,nLevels);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd(:) - coefsActual(:))./abs(coefsExpctd(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
    end
    
end
