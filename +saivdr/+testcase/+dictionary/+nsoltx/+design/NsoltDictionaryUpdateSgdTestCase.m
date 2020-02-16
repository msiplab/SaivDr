classdef NsoltDictionaryUpdateSgdTestCase < matlab.unittest.TestCase
    %NsoltDictionaryUpdateSgdTESTCASE Test case for NsoltDictionaryUpdae
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
        function testDictionaryUpdateDec22Ch44Ord44(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nLevels    = 1;
            nVm = 1;
            %
            srcImgs{1} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = ones(6,8);
            subCoefs{3} = zeros(6,8);
            subCoefs{4} = ones(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = ones(6,8);
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
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            %
            srcImgs{2} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = ones(6,8);
            subCoefs{3} = ones(6,8);
            subCoefs{4} = zeros(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = zeros(6,8);
            subCoefs{8} = ones(6,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{2} = coefs;
            setOfScales{2} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.updater = NsoltDictionaryUpdateSgd(...
                'TrainingImages', srcImgs,...
                'GradObj','on',...
                'IsOptimizationOfMus',false);
            
            % Evaluation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            
            import matlab.unittest.constraints.IsLessThanOrEqualTo;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            %
            options = optimset(...
                'MaxIter',10,...
                'TolX',1e-4,...
                'Display',testCase.display);
            if strcmp(testCase.display,'iter')
                options = optimset(options,'PlotFcns',@optimplotfval);
            end
            
            [~,costPst] = step(testCase.updater,lppufb,options);
            
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre));
            
        end
        
        % Test for Constant step
        function testDictionaryUpdateDec22Ch44Ord44Constant(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nLevels    = 1;
            nVm = 1;
            stepMode = 'Constant';
            %
            srcImgs{1} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = ones(6,8);
            subCoefs{3} = zeros(6,8);
            subCoefs{4} = ones(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = ones(6,8);
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
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            %
            srcImgs{2} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = ones(6,8);
            subCoefs{3} = ones(6,8);
            subCoefs{4} = zeros(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = zeros(6,8);
            subCoefs{8} = ones(6,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{2} = coefs;
            setOfScales{2} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.updater = NsoltDictionaryUpdateSgd(...
                'TrainingImages', srcImgs,...
                'GradObj','on',...
                'Step',stepMode,...
                'IsOptimizationOfMus',false);
            
            % Evaluation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            
            import matlab.unittest.constraints.IsLessThan;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            %
            options = optimset(...
                'MaxIter',10,...
                'TolX',1e-4,...
                'Display',testCase.display);
            if strcmp(testCase.display,'iter')
                options = optimset(options,'PlotFcns',@optimplotfval);
            end
            
            [~,costPst] = step(testCase.updater,lppufb,options);
            
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        % Test for Reciprolac step control
        function testDictionaryUpdateDec22Ch44Ord44Reciprocal(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nLevels    = 1;
            nVm = 1;
            stepMode = 'Reciprocal';
            %
            srcImgs{1} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = ones(6,8);
            subCoefs{3} = zeros(6,8);
            subCoefs{4} = ones(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = ones(6,8);
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
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            %
            srcImgs{2} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = ones(6,8);
            subCoefs{3} = ones(6,8);
            subCoefs{4} = zeros(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = zeros(6,8);
            subCoefs{8} = ones(6,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{2} = coefs;
            setOfScales{2} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.updater = NsoltDictionaryUpdateSgd(...
                'TrainingImages', srcImgs,...
                'GradObj','on',...
                'Step',stepMode,...
                'IsOptimizationOfMus',false);
            
            % Evaluation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            
            import matlab.unittest.constraints.IsLessThan;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            %
            options = optimset(...
                'MaxIter',10,...
                'TolX',1e-4,...
                'Display',testCase.display);
            if strcmp(testCase.display,'iter')
                options = optimset(options,'PlotFcns',@optimplotfval);
            end
            
            [~,costPst] = step(testCase.updater,lppufb,options);
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        % Test for Exponential step control
        function testDictionaryUpdateDec22Ch44Ord44Exponential(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nLevels    = 1;
            nVm = 1;
            stepMode = 'Exponential';
            %
            srcImgs{1} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = ones(6,8);
            subCoefs{3} = zeros(6,8);
            subCoefs{4} = ones(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = ones(6,8);
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
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            %
            srcImgs{2} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = ones(6,8);
            subCoefs{3} = ones(6,8);
            subCoefs{4} = zeros(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = zeros(6,8);
            subCoefs{8} = ones(6,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{2} = coefs;
            setOfScales{2} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.updater = NsoltDictionaryUpdateSgd(...
                'TrainingImages', srcImgs,...
                'GradObj','on',...
                'Step',stepMode,...
                'IsOptimizationOfMus',false);
            
            % Evaluation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            
            import matlab.unittest.constraints.IsLessThan;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            %
            options = optimset(...
                'MaxIter',10,...
                'TolX',1e-4,...
                'Display',testCase.display);
            if strcmp(testCase.display,'iter')
                options = optimset(options,'PlotFcns',@optimplotfval);
            end
            
            [~,costPst] = step(testCase.updater,lppufb,options);
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        % Test for AdaGrad step control
        function testDictionaryUpdateDec22Ch44Ord44AdaGrad(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nLevels    = 1;
            nVm = 1;
            stepMode = 'AdaGrad';
            %
            srcImgs{1} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = ones(6,8);
            subCoefs{3} = zeros(6,8);
            subCoefs{4} = ones(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = ones(6,8);
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
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            %
            srcImgs{2} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = ones(6,8);
            subCoefs{3} = ones(6,8);
            subCoefs{4} = zeros(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = zeros(6,8);
            subCoefs{8} = ones(6,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{2} = coefs;
            setOfScales{2} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.updater = NsoltDictionaryUpdateSgd(...
                'TrainingImages', srcImgs,...
                'GradObj','on',...
                'Step',stepMode,...
                'IsOptimizationOfMus',false);
            
            % Evaluation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            
            import matlab.unittest.constraints.IsLessThan;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            %
            options = optimset(...
                'MaxIter',10,...
                'TolX',1e-4,...
                'Display',testCase.display);
            if strcmp(testCase.display,'iter')
                options = optimset(options,'PlotFcns',@optimplotfval);
            end
            
            [~,costPst] = step(testCase.updater,lppufb,options);
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        
        
        % Test
        function testDictionaryUpdateDec22Ch44Ord22Lv1OptMus(testCase)
            
            isGaAvailable = license('checkout','gads_toolbox');
            if ~isGaAvailable ...
                    || exist('ga','file') ~= 2
                testCase.fail(testCase,'Skipped\n\tGA is not available. ... ');
                return
            end
            
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 2 2 ];
            nLevels = 1;
            nVm = 1;
            %
            srcImgs{1} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = ones(6,8);
            subCoefs{3} = zeros(6,8);
            subCoefs{4} = ones(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = ones(6,8);
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
            sprsCoefs{1} = coefs;
            setOfScales{1} = scales;
            
            %
            srcImgs{2} = rand(12,16);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = ones(6,8);
            subCoefs{3} = ones(6,8);
            subCoefs{4} = zeros(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = zeros(6,8);
            subCoefs{8} = ones(6,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            sprsCoefs{2} = coefs;
            setOfScales{2} = scales;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.updater = NsoltDictionaryUpdateSgd(...
                'TrainingImages', srcImgs,...
                'GradObj','on',...
                'GenerationFactorForMus',2,...
                'Step','Constant',...
                'IsOptimizationOfMus',true);
            
            % Evaluation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            
            import matlab.unittest.constraints.IsLessThanOrEqualTo;
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufb,sprsCoefs,setOfScales);
            %
            set(testCase.updater,...
                'SparseCoefficients',sprsCoefs,...
                'SetOfScales',setOfScales);
            %
            options = gaoptimset('ga');
            options = gaoptimset(options,'Display',testCase.display);
            options = gaoptimset(options,'PopulationSize',4);
            options = gaoptimset(options,'EliteCount',2);
            options = gaoptimset(options,'Generations',2);
            options = gaoptimset(options,'StallGenLimit',16);
            options = gaoptimset(options,'UseParallel','always');
            if strcmp(testCase.display,'iter')
                options = gaoptimset(options,'PlotFcn',@gaplotbestf);
            end
            %
            options = optimset(options,...
                'MaxIter',10,...
                'TolX',1e-4);
            [~,costPst] = step(testCase.updater,lppufb,options);
            
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre));
            
        end
    end
end