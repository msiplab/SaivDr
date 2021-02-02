classdef NsoltDictionaryUpdateGaFminTestCase < matlab.unittest.TestCase
    %NsoltDictionaryUpdateGaFminTESTCASE Test case for
    %NsoltDictionaryUpdateGaFmin
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2020, Shogo MURAMATSU
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
        updater
        display = 'off'
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.updater);
        end
    end
    
    methods (Test)
        
        % Test for default construction
        function testDictionaryUpdateDec22Ch6plus2Ord44(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 6 2 ];
            nOrds = [ 4 4 ];
            nLevels    = 3;
            nVm = 1;
            srcImgs{1} = rand(16,16);
            subCoefs{1} = ones(2,2);
            subCoefs{2} = eye(2);
            subCoefs{3} = zeros(2,2);
            subCoefs{4} = zeros(2,2);
            subCoefs{5} = zeros(2,2);
            subCoefs{6} = zeros(2,2);
            subCoefs{7} = zeros(2,2);
            subCoefs{8} = zeros(2,2);
            subCoefs{9} = zeros(4,4);
            subCoefs{10} = zeros(4,4);
            subCoefs{11} = zeros(4,4);
            subCoefs{12} = zeros(4,4);
            subCoefs{13} = zeros(4,4);
            subCoefs{14} = zeros(4,4);
            subCoefs{15} = zeros(4,4);
            subCoefs{16} = zeros(8,8);
            subCoefs{17} = zeros(8,8);
            subCoefs{18} = zeros(8,8);
            subCoefs{19} = zeros(8,8);
            subCoefs{20} = zeros(8,8);
            subCoefs{21} = zeros(8,8);
            subCoefs{22} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            import saivdr.dictionary.nsoltx.*
            testCase.updater = NsoltDictionaryUpdateGaFmin(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsOptimizationOfMus',false);
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            options = optimoptions('fminunc');
            options = optimoptions(options,'Algorithm','quasi-newton');
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'MaxIter',2);
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            [~,costPst] = step(testCase.updater,lppufb,options);
            
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        function testSetIsFixedCoefs(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 6 2 ];
            nOrds = [ 4 4 ];
            nLevels    = 3;
            nVm = 1;
            srcImgs{1} = rand(16,16);
            subCoefs{1} = ones(2,2);
            subCoefs{2} = eye(2);
            subCoefs{3} = zeros(2,2);
            subCoefs{4} = zeros(2,2);
            subCoefs{5} = zeros(2,2);
            subCoefs{6} = zeros(2,2);
            subCoefs{7} = zeros(2,2);
            subCoefs{8} = zeros(2,2);
            subCoefs{9} = zeros(4,4);
            subCoefs{10} = zeros(4,4);
            subCoefs{11} = zeros(4,4);
            subCoefs{12} = zeros(4,4);
            subCoefs{13} = zeros(4,4);
            subCoefs{14} = zeros(4,4);
            subCoefs{15} = zeros(4,4);
            subCoefs{16} = zeros(8,8);
            subCoefs{17} = zeros(8,8);
            subCoefs{18} = zeros(8,8);
            subCoefs{19} = zeros(8,8);
            subCoefs{20} = zeros(8,8);
            subCoefs{21} = zeros(8,8);
            subCoefs{22} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            import saivdr.dictionary.nsoltx.*
            testCase.updater = NsoltDictionaryUpdateGaFmin(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',true,...
                'IsOptimizationOfMus',false);
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            
            %
            options = optimoptions('fminunc');
            options = optimoptions(options,'Algorithm','quasi-newton');
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'MaxIter',2);
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            
            %
            clnUpdaterFixed = clone(testCase.updater);
            set(clnUpdaterFixed,'IsFixedCoefs',true);
            clnUpdaterUnfixed = clone(testCase.updater);
            set(clnUpdaterUnfixed,'IsFixedCoefs',false);
            
            [~,costOrg]     = step(testCase.updater,lppufb,options);
            [~,costFixed]   = step(clnUpdaterFixed,lppufb,options);
            [~,costUnfixed] = step(clnUpdaterUnfixed,lppufb,options);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(costFixed - costOrg);
            testCase.verifyThat(diff, IsLessThan(1e-6),...
                sprintf('%g',diff));
            diff = norm(costUnfixed - costOrg);
            testCase.verifyThat(diff, IsGreaterThan(1e-6),...
                sprintf('%g',diff));
            
        end
        
        function testDictionaryUpdateDec22Ch5plus3Ord44Ga(testCase)
            
            isGaAvailable = license('checkout','gads_toolbox');
            if ~isGaAvailable ...
                    || exist('ga','file') ~= 2
                testCase.fail(testCase,'Skipped\n\tGA is not available. ... ');
                return
            end
            
            nDecs = [ 2 2 ];
            nChs  = [ 5 3 ];
            nOrds = [ 4 4 ];
            nLevels    = 3;
            nVm = 1;
            optfcn = @ga;
            srcImgs{1} = rand(16,16);
            subCoefs{1} = ones(2,2);
            subCoefs{2} = eye(2);
            subCoefs{3} = zeros(2,2);
            subCoefs{4} = zeros(2,2);
            subCoefs{5} = zeros(2,2);
            subCoefs{6} = zeros(2,2);
            subCoefs{7} = zeros(2,2);
            subCoefs{8} = zeros(2,2);
            subCoefs{9} = zeros(4,4);
            subCoefs{10} = zeros(4,4);
            subCoefs{11} = zeros(4,4);
            subCoefs{12} = zeros(4,4);
            subCoefs{13} = zeros(4,4);
            subCoefs{14} = zeros(4,4);
            subCoefs{15} = zeros(4,4);
            subCoefs{16} = zeros(8,8);
            subCoefs{17} = zeros(8,8);
            subCoefs{18} = zeros(8,8);
            subCoefs{19} = zeros(8,8);
            subCoefs{20} = zeros(8,8);
            subCoefs{21} = zeros(8,8);
            subCoefs{22} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            import saivdr.dictionary.nsoltx.*
            testCase.updater = NsoltDictionaryUpdateGaFmin(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'OptimizationFunction',optfcn,...
                'MaxIterOfHybridFmin',2,...
                'IsOptimizationOfMus',false);
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            options = gaoptimset('ga');
            options = gaoptimset(options,'Display',testCase.display);
            options = gaoptimset(options,'PopulationSize',4);
            options = gaoptimset(options,'EliteCount',2);
            popInitRange = [angs(:)-pi angs(:)+pi].';
            options = gaoptimset(options,'PopInitRange',popInitRange);
            options = gaoptimset(options,'Generations',2);
            options = gaoptimset(options,'StallGenLimit',4);
            options = gaoptimset(options,'UseParallel','always');
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            [~, costPst] = step(testCase.updater,lppufb,options);
            
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
                
        function testDictionaryUpdateDec22Ch44Ord44Lv1GaGradObj(testCase)
            
            isGaAvailable = license('checkout','gads_toolbox');
            if ~isGaAvailable ...
                    || exist('ga','file') ~= 2
                testCase.fail(testCase,'Skipped\n\tGA is not available. ... ');
                return
            end
            
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nLevels    = 1;
            nVm = 1;
            optfcn = @ga;
            srcImgs{1} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = eye(6,8);
            subCoefs{3} = zeros(6,8);
            subCoefs{4} = zeros(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8); 
            subCoefs{7} = zeros(6,8); 
            subCoefs{8} = zeros(6,8); 
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{1}   = coefs;
            setOfScales{1} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            import saivdr.dictionary.nsoltx.*
            testCase.updater = NsoltDictionaryUpdateGaFmin(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels', nLevels,...
                'OptimizationFunction', optfcn,...
                'MaxIterOfHybridFmin', 2,...
                'GenerationFactorForMus', 2,...
                'GradObj','on',...
                'IsOptimizationOfMus',true);
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            options = gaoptimset('ga');
            options = gaoptimset(options,'Display',testCase.display);
            options = gaoptimset(options,'PopulationSize',4);
            options = gaoptimset(options,'EliteCount',2);
            popInitRange = [angs(:)-pi angs(:)+pi].';
            options = gaoptimset(options,'PopInitRange',popInitRange);
            options = gaoptimset(options,'Generations',2);
            options = gaoptimset(options,'StallGenLimit',4);
            options = gaoptimset(options,'UseParallel','always');
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            [~, costPst] = step(testCase.updater,lppufb,options);
            
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        function testDictionaryUpdateDec22Ch4plus4Ord44GaOptMus(testCase)
            
            isGaAvailable = license('checkout','gads_toolbox');
            if ~isGaAvailable ...
                    || exist('ga','file') ~= 2
                testCase.fail(testCase,'Skipped\n\tGA is not available. ... ');
                return
            end
            
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nLevels    = 3;
            nVm = 1;
            optfcn = @ga;
            srcImgs{1} = rand(16,16);
            subCoefs{1} = ones(2,2);
            subCoefs{2} = eye(2);
            subCoefs{3} = zeros(2,2);
            subCoefs{4} = zeros(2,2);
            subCoefs{5} = zeros(2,2);
            subCoefs{6} = zeros(2,2);
            subCoefs{7} = zeros(2,2);
            subCoefs{8} = zeros(2,2);
            subCoefs{9} = zeros(4,4);
            subCoefs{10} = zeros(4,4);
            subCoefs{11} = zeros(4,4);
            subCoefs{12} = zeros(4,4);
            subCoefs{13} = zeros(4,4);
            subCoefs{14} = zeros(4,4);
            subCoefs{15} = zeros(4,4);
            subCoefs{16} = zeros(8,8);
            subCoefs{17} = zeros(8,8);
            subCoefs{18} = zeros(8,8);
            subCoefs{19} = zeros(8,8);
            subCoefs{20} = zeros(8,8);
            subCoefs{21} = zeros(8,8);
            subCoefs{22} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            import saivdr.dictionary.nsoltx.*
            testCase.updater = NsoltDictionaryUpdateGaFmin(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'OptimizationFunction',optfcn,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2,...
                'IsOptimizationOfMus',true);
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            options = gaoptimset('ga');
            options = gaoptimset(options,'Display',testCase.display);
            options = gaoptimset(options,'PopulationSize',4);
            options = gaoptimset(options,'EliteCount',2);
            popInitRange = [angs(:)-pi angs(:)+pi].';
            options = gaoptimset(options,'PopInitRange',popInitRange);
            options = gaoptimset(options,'Generations',2);
            options = gaoptimset(options,'StallGenLimit',4);
            options = gaoptimset(options,'UseParallel','always');
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            [~, costPst]= step(testCase.updater,lppufb,options);
            
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        % Test
        function testDictionaryUpdateDec222Ch55Ord222(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 2 ];
            nLevels    = 3;
            nVm = 1;
            srcImgs{1} = rand(16,16,16);
            subCoefs{1} = ones(2,2,2);
            subCoefs{2} = ones(2,2,2);
            subCoefs{3} = zeros(2,2,2);
            subCoefs{4} = zeros(2,2,2);
            subCoefs{5} = zeros(2,2,2);
            subCoefs{6} = zeros(2,2,2);
            subCoefs{7} = zeros(2,2,2);
            subCoefs{8} = zeros(2,2,2);
            subCoefs{9} = zeros(2,2,2);
            subCoefs{10} = zeros(2,2,2);
            subCoefs{11} = zeros(4,4,4);
            subCoefs{12} = zeros(4,4,4);
            subCoefs{13} = zeros(4,4,4);
            subCoefs{14} = zeros(4,4,4);
            subCoefs{15} = zeros(4,4,4);
            subCoefs{16} = zeros(4,4,4);
            subCoefs{17} = zeros(4,4,4);
            subCoefs{18} = zeros(4,4,4);
            subCoefs{19} = zeros(4,4,4);
            subCoefs{20} = zeros(8,8,8);
            subCoefs{21} = zeros(8,8,8);
            subCoefs{22} = zeros(8,8,8);
            subCoefs{23} = zeros(8,8,8);
            subCoefs{24} = zeros(8,8,8);
            subCoefs{25} = zeros(8,8,8);
            subCoefs{26} = zeros(8,8,8);
            subCoefs{27} = zeros(8,8,8);
            subCoefs{28} = zeros(8,8,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            import saivdr.dictionary.nsoltx.*
            testCase.updater = NsoltDictionaryUpdateGaFmin(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsOptimizationOfMus',false);
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            options = optimoptions('fminunc');
            options = optimoptions(options,'Algorithm','quasi-newton');
            options = optimoptions(options,'Display',testCase.display);
            options = optimoptions(options,'MaxIter',2);
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            [~,costPst] = step(testCase.updater,lppufb,options);
            
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        function testDictionaryUpdateDec222Ch54Ord222Ga(testCase)
            
            isGaAvailable = license('checkout','gads_toolbox');
            if ~isGaAvailable ...
                    || exist('ga','file') ~= 2
                testCase.fail(testCase,'Skipped\n\tGA is not available. ... ');
                return
            end
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 2 2 2 ];
            nLevels    = 3;
            nVm = 1;
            optfcn = @ga;
            srcImgs{1} = rand(16,16,16);
            subCoefs{1} = ones(2,2,2);
            subCoefs{2} = ones(2,2,2);
            subCoefs{3} = zeros(2,2,2);
            subCoefs{4} = zeros(2,2,2);
            subCoefs{5} = zeros(2,2,2);
            subCoefs{6} = zeros(2,2,2);
            subCoefs{7} = zeros(2,2,2);
            subCoefs{8} = zeros(2,2,2);
            subCoefs{9} = zeros(2,2,2);
            subCoefs{10} = zeros(4,4,4);
            subCoefs{11} = zeros(4,4,4);
            subCoefs{12} = zeros(4,4,4);
            subCoefs{13} = zeros(4,4,4);
            subCoefs{14} = zeros(4,4,4);
            subCoefs{15} = zeros(4,4,4);
            subCoefs{16} = zeros(4,4,4);
            subCoefs{17} = zeros(4,4,4);
            subCoefs{18} = zeros(8,8,8);
            subCoefs{19} = zeros(8,8,8);
            subCoefs{20} = zeros(8,8,8);
            subCoefs{21} = zeros(8,8,8);
            subCoefs{22} = zeros(8,8,8);
            subCoefs{23} = zeros(8,8,8);
            subCoefs{24} = zeros(8,8,8);
            subCoefs{25} = zeros(8,8,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            import saivdr.dictionary.nsoltx.*
            testCase.updater = NsoltDictionaryUpdateGaFmin(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'OptimizationFunction',optfcn,...
                'MaxIterOfHybridFmin',2,...
                'IsOptimizationOfMus',false);
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            options = gaoptimset('ga');
            options = gaoptimset(options,'Display',testCase.display);
            options = gaoptimset(options,'PopulationSize',4);
            options = gaoptimset(options,'EliteCount',2);
            popInitRange = [angs(:)-pi angs(:)+pi].';
            options = gaoptimset(options,'PopInitRange',popInitRange);
            options = gaoptimset(options,'Generations',2);
            options = gaoptimset(options,'StallGenLimit',4);
            options = gaoptimset(options,'UseParallel','always');
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            [~, costPst] = step(testCase.updater,lppufb,options);
            
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        
        function testDictionaryUpdateDec222Ch64Ord222GaOptMus(testCase)
            
            isGaAvailable = license('checkout','gads_toolbox');
            if ~isGaAvailable ...
                    || exist('ga','file') ~= 2
                testCase.fail(testCase,'Skipped\n\tGA is not available. ... ');
                return
            end
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 4 ];
            nOrds = [ 2 2 2 ];
            nLevels    = 3;
            nVm = 1;
            optfcn = @ga;
            srcImgs{1} = rand(16,16,16);
            subCoefs{1} = ones(2,2,2);
            subCoefs{2} = ones(2,2,2);
            subCoefs{3} = zeros(2,2,2);
            subCoefs{4} = zeros(2,2,2);
            subCoefs{5} = zeros(2,2,2);
            subCoefs{6} = zeros(2,2,2);
            subCoefs{7} = zeros(2,2,2);
            subCoefs{8} = zeros(2,2,2);
            subCoefs{9} = zeros(2,2,2);
            subCoefs{10} = zeros(2,2,2);
            subCoefs{11} = zeros(4,4,4);
            subCoefs{12} = zeros(4,4,4);
            subCoefs{13} = zeros(4,4,4);
            subCoefs{14} = zeros(4,4,4);
            subCoefs{15} = zeros(4,4,4);
            subCoefs{16} = zeros(4,4,4);
            subCoefs{17} = zeros(4,4,4);
            subCoefs{18} = zeros(4,4,4);
            subCoefs{19} = zeros(4,4,4);
            subCoefs{20} = zeros(8,8,8);
            subCoefs{21} = zeros(8,8,8);
            subCoefs{22} = zeros(8,8,8);
            subCoefs{23} = zeros(8,8,8);
            subCoefs{24} = zeros(8,8,8);
            subCoefs{25} = zeros(8,8,8);
            subCoefs{26} = zeros(8,8,8);
            subCoefs{27} = zeros(8,8,8);
            subCoefs{28} = zeros(8,8,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            import saivdr.dictionary.nsoltx.*
            testCase.updater = NsoltDictionaryUpdateGaFmin(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'OptimizationFunction',optfcn,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2,...
                'IsOptimizationOfMus',true);
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            options = gaoptimset('ga');
            options = gaoptimset(options,'Display',testCase.display);
            options = gaoptimset(options,'PopulationSize',4);
            options = gaoptimset(options,'EliteCount',2);
            popInitRange = [angs(:)-pi angs(:)+pi].';
            options = gaoptimset(options,'PopInitRange',popInitRange);
            options = gaoptimset(options,'Generations',2);
            options = gaoptimset(options,'StallGenLimit',4);
            options = gaoptimset(options,'UseParallel','always');
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            [~, costPst]= step(testCase.updater,lppufb,options);
            
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        function testDictionaryUpdateDec112Ch22Ord222GaOptMus(testCase)
            
            isGaAvailable = license('checkout','gads_toolbox');
            if ~isGaAvailable ...
                    || exist('ga','file') ~= 2
                testCase.fail(testCase,'Skipped\n\tGA is not available. ... ');
                return
            end
            
            nDecs = [ 1 1 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 2 ];
            nLevels    = 3;
            nVm = 0;
            optfcn = @ga;
            srcImgs{1} = rand(16,16,16);
            subCoefs{1} = ones(16,16,2);
            subCoefs{2} = ones(16,16,2);
            subCoefs{3} = zeros(16,16,2);
            subCoefs{4} = zeros(16,16,2);
            subCoefs{5} = zeros(16,16,4);
            subCoefs{6} = zeros(16,16,4);
            subCoefs{7} = zeros(16,16,4);
            subCoefs{8} = zeros(16,16,8);
            subCoefs{9} = zeros(16,16,8);
            subCoefs{10} = zeros(16,16,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            import saivdr.dictionary.nsoltx.*
            testCase.updater = NsoltDictionaryUpdateGaFmin(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'OptimizationFunction',optfcn,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2,...
                'IsOptimizationOfMus',true);
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = pi*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            options = gaoptimset('ga');
            options = gaoptimset(options,'Display',testCase.display);
            options = gaoptimset(options,'PopulationSize',4);
            options = gaoptimset(options,'EliteCount',2);
            popInitRange = [angs(:)-pi angs(:)+pi].';
            options = gaoptimset(options,'PopInitRange',popInitRange);
            options = gaoptimset(options,'Generations',2);
            options = gaoptimset(options,'StallGenLimit',4);
            options = gaoptimset(options,'UseParallel','always');
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            [~, costPst]= step(testCase.updater,lppufb,options);
            
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
    end
end

