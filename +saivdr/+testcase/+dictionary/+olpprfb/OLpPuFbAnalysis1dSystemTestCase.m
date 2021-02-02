classdef OLpPuFbAnalysis1dSystemTestCase < matlab.unittest.TestCase
    %OLPPUFBANALYSIS1DSYSTEMTESTCASE Test case for OLpPuFbAnalysis1dSystem
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
        function testDefaultConstructionTypeI(testCase)
            
            % Expected values
            import saivdr.dictionary.olpprfb.*
            lppufbExpctd = OvsdLpPuFb1dTypeIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            testCase.analyzer = OLpPuFbAnalysis1dSystem();
            
            % Actual value
            lppufbActual = get(testCase.analyzer,'LpPuFb1d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
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
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'NumberOfSymmetricChannels',nChs(ChannelGroup.UPPER),...
                'NumberOfAntisymmetricChannels',nChs(ChannelGroup.LOWER));
            
            % Actual value
            lppufbActual = get(testCase.analyzer,'LpPuFb1d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end        

        % Test
        function testStepDec1Ch22Ord0Level1Vm0(testCase)
            
            dec = 1;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
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
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = nLen/dec;
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                subCoef = downsample(...
                    cconv(srcSeq.',step(lppufb,[],[],iSubband),nLen),...
                    dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./dec,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end
    
        % Test
        function testStepDec1Ch22Ord00Level1Vm1(testCase)
            
            dec = 1;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
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
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = nLen/dec;
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                subCoef = downsample(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./dec,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));

        end

        % Test
        function testStepDec1Ch22Ord0Level2PeriodicExtVm0(testCase)
            
            dec = 1;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    nLen/dec),dec);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};
            coefs{7} = coefsExpctdLv1{4};            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));  
            
        end

        % Test
        function testStepDec1Ch22Ord0Level2PeriodicExtVm1(testCase)
            
            dec = 1;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...,...
                    nLen),dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    nLen/dec),dec);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};
            coefs{7} = coefsExpctdLv1{4};            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));  
            
        end

        % Test
        function testStepDec2Ch22Ord0Level1PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
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
           set(lppufb,'OutputMode','AnalysisFilterAt');
           nSubCoefs = nLen/dec;
           coefsExpctd = zeros(1,ch*nSubCoefs);
           phs = dec-1;
           for iSubband = 1:ch
                subCoef = downsample(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...,...
                    nLen),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./dec,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual))./abs(coefsExpctd);
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch22Ord0Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    nLen/dec),dec,phs);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};
            coefs{7} = coefsExpctdLv1{4};            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));  

        end

        % Test
        function testStepDec2Ch33Ord0Level1(testCase)
            
            dec = 2;
            nChs = [ 3 3 ];
            ch = sum(nChs);
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
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
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = nLen/dec;
            coefsExpctd = zeros(1,ch*nSubCoefs);
            phs = dec-1;
            for iSubband = 1:ch
                subCoef = downsample(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...,...
                    nLen),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./dec,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
        end

        % Test
        function testIterDecompDec2Ch33Ord0Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 3 3 ];
            ch = sum(nChs);
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...,...
                    nLen),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    nLen/dec),dec,phs);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv1{2};
            coefs{8} = coefsExpctdLv1{3};
            coefs{9} = coefsExpctdLv1{4};            
            coefs{10} = coefsExpctdLv1{5};
            coefs{11} = coefsExpctdLv1{6};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));  
            
        end

       % Test
        function testStepDec2Ch44Ord0Level1PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 4 4 ];
            ch = sum(nChs);
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
          % Expected values
          release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = nLen/dec;
            coefsExpctd = zeros(1,ch*nSubCoefs);
            phs = dec-1;
            for iSubband = 1:ch
                subCoef = downsample(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...,...
                    nLen),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./dec,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch44Ord0Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 4 4 ];
            ch = sum(nChs);
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
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
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...,...
                    nLen),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    nLen/dec),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));            
            
        end

        % Test
        function testStepDec1Ch22Ord2Level1(testCase)
            
            dec = 1;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
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
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = nLen/dec;
            coefsExpctd = zeros(1,ch*nSubCoefs);
            phs = dec-1;
            offset = -dec*ord/2;
            for iSubband = 1:ch
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./dec,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

        % Test
        function testStepDec1Ch22Ord2Level2PeriodicExt(testCase)
            
            dec = 1;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
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
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            phs = dec-1;
            offset = -dec*ord/2;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    nLen/dec),offset),dec,phs);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};
            coefs{7} = coefsExpctdLv1{4};            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));            
        end

        % Test
        function testStepDec2Ch22Ord2Level1PeridicExt(testCase)
            
            dec = 2;
            chs = [ 2 2 ];
            nChs = sum(chs);
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = nLen/dec;
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            phs = dec-1;
            offset = -dec*ord/2;            
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./dec,nChs,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',chs(1),...
                'NumberOfAntisymmetricChannels',chs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec2Ch22Ord0Level1PeridicExt(testCase)
            
            dec = 2;
            chs = [ 2 2 ];
            nChs = sum(chs);
            ord = 0;
            nLen = 64;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = nLen/dec;
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            phs = dec-1;
            offset = -dec*ord/2;                        
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./dec,nChs,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',chs(1),...
                'NumberOfAntisymmetricChannels',chs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch22Ord2Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 2;
            nLen = 64;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
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
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            phs = dec-1;
            offset = -dec*ord/2;                                    
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...,...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    nLen/dec),offset),dec,phs);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};
            coefs{7} = coefsExpctdLv1{4};            
            nSubbands = length(coefs);            
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
            
        end

        % Test
        function testStepDec2Ch33Ord2Level1(testCase)
            
            dec = 2;
            nChs = [ 3 3 ];
            ch = sum(nChs);
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = nLen/dec;
            coefsExpctd = zeros(1,ch*nSubCoefs);
            phs = dec-1;
            offset = -dec*ord/2;                                                
            for iSubband = 1:ch
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...,...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./dec,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch33Ord2Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 3 3 ];            
            ch = sum(nChs);
            ord = 2;
            nLen = 64;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
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
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            phs = dec-1;
            offset = -dec*ord/2;                                                            
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...,...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    nLen/dec),offset),dec,phs);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv1{2};
            coefs{8} = coefsExpctdLv1{3};
            coefs{9} = coefsExpctdLv1{4};            
            coefs{10} = coefsExpctdLv1{5};
            coefs{11} = coefsExpctdLv1{6};
            nSubbands = length(coefs);            
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec2Ch44Ord2Level1(testCase)
            
            dec = 2;
            nChs = [ 4 4 ];
            ch = sum(nChs);
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
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
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = nLen/dec;
            coefsExpctd = zeros(1,ch*nSubCoefs);
            phs = dec-1;
            offset = -dec*ord/2;                                                                        
            for iSubband = 1:ch
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./dec,ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
            
        end

        % Test
        function testStepDec2Ch44Ord2Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 4 4 ];
            ch = sum(nChs);
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
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
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            phs = dec-1;
            offset = -dec*ord/2;                                                                                    
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...,...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    nLen/dec),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
                        
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');

            % Actual values
            [coefsActual, scalesActual]= step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));            
        end

        % Level 3, dec 22 ch 4 order 4 
        function testStepDec2Ch44Ord4Level3PeriodicExt(testCase)
            
            dec = 2;
            nChs =  [ 4 4 ];
            ch = sum(nChs);
            ord = 4;
            nLen = 64;
            srcSeq = rand(1,nLen);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            phs = dec-1;
            offset = -dec*ord/2;               
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...,...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    nLen/dec),offset),dec,phs);
            end      
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv3{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv2{1},...
                    step(lppufb,[],[],iSubband),...,...
                    nLen/(dec^2)),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end        

        % Level 3, dec 2 ch 8  order 44 
        function testSetLpPuFb1dDec2Ch44Ord4(testCase)
            
            dec = 2;
            ch =  [ 4 4 ];
            ord = 4;
            nLen = 64;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcSeq);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs); 
            coefsPst1 = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation 
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));            
        end
        
       % Level 1, dec 4 ch 8  order 2 
        function testSetLpPuFb1dDec4Ch88Ord2(testCase)
            
            dec = 4;
            ch =  [ 8 8 ];
            ord = 2;
            nLen = 64;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcSeq);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs); 
            coefsPst1 = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation 
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));            
        end

        % Level 3, dec 2 ch 44  order 4
        function testIsCloneFalse(testCase)
            
            dec = 2;
            ch =  [ 4 4 ];
            ord = 4;
            nLen = 64;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination',...
                'IsCloneLpPuFb1d',true);
            
            % Pre
            coefsPre1 = step(testCase.analyzer,srcSeq);
            
            % Pst
            angs = randn(size(get(lppufb,'Angles')));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre1(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,sprintf('%g',diff));
            
            % Instantiation of target class
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'BoundaryOperation','Termination',...
                'IsCloneLpPuFb1d',false);
            
            % Pre
            coefsPre1 = step(testCase.analyzer,srcSeq);
            
            % Pst
            angs = randn(size(get(lppufb,'Angles')));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst1(:)-coefsPre1(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));            
           
        end

        % Test
        function testClone(testCase)
            
            dec = 2;
            ch =  [ 4 4 ];
            ord = 4;
            nLen = 64;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                
                'BoundaryOperation','Termination');
            %s = matlab.System.saveObject(testCase.analyzer);
            
            % Clone
            cloneAnalyzer = clone(testCase.analyzer);
            
            % Evaluation
            prpOrg = get(testCase.analyzer,'LpPuFb1d');
            prpCln = get(cloneAnalyzer,'LpPuFb1d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            testCase.verifyEqual(cloneAnalyzer,testCase.analyzer);
            testCase.verifyFalse(cloneAnalyzer == testCase.analyzer);            
            %
            [coefExpctd,scaleExpctd] = step(testCase.analyzer,srcSeq);
            [coefActual,scaleActual] = step(cloneAnalyzer,srcSeq);
            testCase.assertEqual(coefActual,coefExpctd);
            testCase.assertEqual(scaleActual,scaleExpctd);
            
        end     


       % Test
        function testDefaultConstructionTypeII(testCase)
            
            % Expected values
            import saivdr.dictionary.olpprfb.*
            lppufbExpctd = OvsdLpPuFb1dTypeIIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufbExpctd);
            
            % Actual value
            lppufbActual = get(testCase.analyzer,'LpPuFb1d');
            
            % Evaluation
            testCase.verifyEqual(lppufbActual,lppufbExpctd);
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
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'NumberOfSymmetricChannels',nChs(ChannelGroup.UPPER),...
                'NumberOfAntisymmetricChannels',nChs(ChannelGroup.LOWER));
            
            % Actual value
            lppufbActual = get(testCase.analyzer,'LpPuFb1d');
            
            % Evaluation
            testCase.verifyEqual(lppufbActual,lppufbExpctd);
        end
                 
        % Test
        function testStepDec1Ch32Ord0Level1PeriodicExtVm0(testCase)
            
            dec = 1;
            decch = [ dec 3 2 ];
            nChs = sum(decch(2:3));
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            phs = dec - 1;
            offset = -dec*ord/2;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec1Ch32Ord0Level1PeriodicExtVm1(testCase)
            
            dec = 1;
            decch = [ dec 3 2 ];
            nChs = sum(decch(2:3));
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -dec*ord/2;
            phs = dec-1 ;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testStepDec11Ch32Ord00Level2PeriodicExtVm0(testCase)
            
            dec = 1;
            decch = [ dec 3 2 ];
            nChs = sum(decch(2:3));
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels', decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testStepDec1Ch32Ord0Level2PeriodicExtVm1(testCase)
            
            dec = 1;
            decch = [ dec 3 2 ];
            nChs = sum(decch(2:3));
            ord = 0;
            nLen = 32;
            
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels', decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch32Ord0Level1(testCase)
            
            dec = 2;
            decch = [ dec 3 2 ];
            nChs = sum(decch(2:3));
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec4Ch98Ord0Level1(testCase)
            
            dec = 4;
            decch = [ dec 9 8 ];
            nChs = sum(decch(2:3));
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec4Ch98Ord2Level1(testCase)
            
            dec = 4;
            decch = [ dec 9 8 ];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec2Ch32Ord0Level2eriodicExt(testCase)
            
            dec = 2;
            decch = [ dec 3 2];
            ch = sum(decch(2:3));
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            offset = -dec*ord/2;
            phs    = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec2Ch43Ord0Level1(testCase)
            
            dec = 2;
            decch = [ dec 4 3 ];
            nChs = sum(decch(2:3));
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch43Ord0Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec 4 3 ];
            ch = sum(decch(2:3));
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv1{2};
            coefs{9} = coefsExpctdLv1{3};
            coefs{10} = coefsExpctdLv1{4};
            coefs{11} = coefsExpctdLv1{5};
            coefs{12} = coefsExpctdLv1{6};
            coefs{13} = coefsExpctdLv1{7};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
            
        end
        
        % Test
        function testStepDec2Ch54Ord0Level1(testCase)
            
            dec = 2;
            decch = [ dec 5 4 ];
            nChs= sum(decch(2:3));
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end

        % Test
        function testStepDec2Ch54Ord0Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec 5 4 ];
            ch = sum(decch(2:3));
            ord = 0;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testStepDec1Ch32Ord2Level1(testCase)
            
            dec = 1;
            decch = [ dec 3 2 ];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

        % Test
        function testStepDec1Ch32Ord2Level2PeriodicExt(testCase)
            
            dec = 1;
            decch = [ dec 3 2 ];
            ch = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
            
        end

        % Test
        function testStepDec2Ch32Ord2Level1(testCase)
            
            dec = 2;
            decch = [ dec 3 2 ];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch32Ord2Level2eriodicExt(testCase)
            
            dec = 2;
            decch = [ dec 3 2 ];
            ch = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch43Ord2Level1(testCase)
            
            dec = 2;
            decch = [ dec 4 3 ];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch43Ord2Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec 4 3 ];
            ch = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv1{2};
            coefs{9} = coefsExpctdLv1{3};
            coefs{10} = coefsExpctdLv1{4};
            coefs{11} = coefsExpctdLv1{5};
            coefs{12} = coefsExpctdLv1{6};
            coefs{13} = coefsExpctdLv1{7};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch54Ord2Level1(testCase)
            
            dec = 2;
            decch = [ dec 5 4 ];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels', decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
            
        end

        % Test
        function testSteppDec2Ch54Ord2Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 5 4 ];
            decch = [ dec nChs ];
            ch = sum(nChs);
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Level 3, dec 1 ch 54 order 8
        function testStepDec1Ch54Ord8Level3PeriodicExt(testCase)
            
            dec = 1;
            nChs = [ 5 4 ];
            decch = [ dec nChs ];
            ch = sum(nChs);
            ord = 8;
            nLen = 64;
            srcSeq = rand(1,nLen);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
            end
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv3{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv2{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/(dec^2)),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-6,...
                sprintf('%g',diff));
            
        end

        % Level 3, dec 2 ch 54 order 4
        function testStepDec2Ch54Ord4Level3PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 5 4 ];
            decch = [ dec nChs ];
            ch = sum(nChs);
            ord = 4;
            nLen = 64;
            srcSeq = rand(1,nLen);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
            end
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv3{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv2{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/(dec^2)),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec2Ch32Ord2Level1PeriodicExt(testCase)
            
            dec = 2;
            ch = [3 2];
            nChs = sum(ch);
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(dec);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./dec,nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',ch(1),...
                'NumberOfAntisymmetricChannels',ch(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch32Ord2Lev2PeriodicExt(testCase)
            
            decch = [2 3 2];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            offset = -decch(1)*ord/2;
            phs = decch(1)-1;
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),decch(1),phs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/decch(1)),offset),decch(1),phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
        end

        % Test
        function testStepDec2Ch42Ord2Level1PeriodicExt(testCase)
            
            decch = [2 4 2];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -decch(1)*ord/2;
            phs = decch(1)-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),decch(1),phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch42Ord2Level2PeriodicExt(testCase)
            
            decch = [2 4 2];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            offset = -decch(1)*ord/2;
            phs = decch(1)-1;
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),decch(1),phs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/decch(1)),offset),decch(1),phs);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv1{2};
            coefs{8} = coefsExpctdLv1{3};
            coefs{9} = coefsExpctdLv1{4};
            coefs{10} = coefsExpctdLv1{5};
            coefs{11} = coefsExpctdLv1{6};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testSetLpPuFb1dDec2Ch52Ord4(testCase)
            
            dec = 2;
            ch =  [ 5 2 ];
            ord = 4;
            nLen = 64;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcSeq);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end

        % Test
        function testIsCloneFalseTypeII(testCase)
            
            dec = 2;
            ch =  [ 6 2 ];
            ord = 4;
            nLen = 64;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                    
                'BoundaryOperation','Termination',...
                'IsCloneLpPuFb1d',true);
            
            % Pre
            coefsPre1 = step(testCase.analyzer,srcSeq);
            
            % Pst
            angs = randn(size(get(lppufb,'Angles')));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre1(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,sprintf('%g',diff));
            
            % Instantiation of target class
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                                    
                'BoundaryOperation','Termination',...
                'IsCloneLpPuFb1d',false);
            
            % Pre
            coefsPre1 = step(testCase.analyzer,srcSeq);
            
            % Pst
            angs = randn(size(get(lppufb,'Angles')));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst1(:)-coefsPre1(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));            
           
        end

        % Test
        function testCloneTypeII(testCase)
            
            dec = 2;
            ch =  [ 5 3 ];
            ord = 4;
            nLen = 64;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                                    
                'BoundaryOperation','Termination');
            %s = matlab.System.saveObject(testCase.analyzer);

            % Clone
            cloneAnalyzer = clone(testCase.analyzer);
            
            % Evaluation
            prpOrg = get(testCase.analyzer,'LpPuFb1d');
            prpCln = get(cloneAnalyzer,'LpPuFb1d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            testCase.verifyEqual(cloneAnalyzer,testCase.analyzer);
            testCase.verifyFalse(cloneAnalyzer == testCase.analyzer);            
            %            
            [coefExpctd,scaleExpctd] = step(testCase.analyzer,srcSeq);
            [coefActual,scaleActual] = step(cloneAnalyzer,srcSeq);
            testCase.assertEqual(coefActual,coefExpctd);
            testCase.assertEqual(scaleActual,scaleExpctd);
            
        end
        
        function testStepDec1Ch45Ord8Level3PeriodicExt(testCase)
            
            dec = 1;
            nChs = [ 4 5 ];
            decch = [ dec nChs ];
            ch = sum(nChs);
            ord = 8;
            nLen = 64;
            srcSeq = rand(1,nLen);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
            end
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv3{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv2{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/(dec^2)),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-6,...
                sprintf('%g',diff));
            
        end

        function testStepDec2Ch45Ord4Level3PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 4 5 ];
            decch = [ dec nChs ];
            ch = sum(nChs);
            ord = 4;
            nLen = 64;
            srcSeq = rand(1,nLen);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            offset = -dec*ord/2;
            phs = dec-1;
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),dec,phs);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/dec),offset),dec,phs);
            end
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv3{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv2{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/(dec^2)),offset),dec,phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch23Ord2Level1PeriodicExt(testCase)
            
            decch = [2 2 3];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -decch(1)*ord/2;
            phs = decch(1)-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),decch(1),phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
            
        end

        % Test
        function testStepDec2Ch23Ord2Level2PeriodicExt(testCase)
            
            decch = [2 2 3];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            offset = -decch(1)*ord/2;
            phs = decch(1)-1;
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),decch(1),phs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/decch(1)),offset),decch(1),phs);
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
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
        end

        % Test
        function testStepDec2Ch24Ord2Level1PeriodicExt(testCase)
            
            decch = [2 2 4];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcSeq)/(decch(1));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            offset = -decch(1)*ord/2;
            phs = decch(1)-1;
            for iSubband = 1:nChs
                subCoef = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),decch(1),phs);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(length(srcSeq)./decch(1),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

        % Test
        function testStepDec2Ch24Ord2Level2PeriodicExt(testCase)
            
            decch = [2 2 4];
            nChs = sum(decch(2:3));
            ord = 2;
            nLen = 32;
            srcSeq = rand(1,nLen);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            offset = -decch(1)*ord/2;
            phs = decch(1)-1;
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    circshift(...
                    cconv(srcSeq.',...
                    step(lppufb,[],[],iSubband),...
                    nLen),offset),decch(1),phs);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    circshift(...
                    cconv(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    nLen/decch(1)),offset),decch(1),phs);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv1{2};
            coefs{8} = coefsExpctdLv1{3};
            coefs{9} = coefsExpctdLv1{4};
            coefs{10} = coefsExpctdLv1{5};
            coefs{11} = coefsExpctdLv1{6};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,1);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband) = length(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                                    
                'NumberOfSymmetricChannels',decch(2),...
                'NumberOfAntisymmetricChannels',decch(3),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcSeq);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual)./abs(coefsExpctd));
            testCase.verifyEqual(coefsActual,coefsExpctd,'RelTol',1e-7,...
                sprintf('%g',diff));
            
        end

        % Test
        function testSetLpPuFb1dDec2Ch25Ord4(testCase)
            
            dec = 2;
            ch =  [ 2 5 ];
            ord = 4;
            nLen = 64;
            nLevels = 1;
            srcSeq = rand(1,nLen);
            
            % Preparation
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                                    
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcSeq);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation
            testCase.analyzer = OLpPuFbAnalysis1dSystem(...
                'LpPuFb1d',lppufb,...
                'NumberOfLevels',nLevels,...                                                                    
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcSeq);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end

    end
    
end
