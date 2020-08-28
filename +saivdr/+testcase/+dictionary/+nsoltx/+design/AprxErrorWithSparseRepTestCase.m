classdef AprxErrorWithSparseRepTestCase < matlab.unittest.TestCase
    %APRXERRORWITHSPARSEREPTESTCASE Test case for AprxErrorWithSparseRep
    %
    % Requirements: MATLAB R2015b
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
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    properties
        aprxerr
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.aprxerr);
        end
    end
    
    methods (Test)
                
        % Test
        function testAprxErrDec22Ch44Ord44Vm1Lv1GradObjOn(testCase)
            
            % Parameters
            width  = 16;
            height = 12;
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 1;
            srcImgs{1} = rand(height,width);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = zeros(6,8);
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
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis2dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'GradObj','on',...
                'IsFixedCoefs',true);
            
            % Expected values
            import saivdr.testcase.dictionary.nsoltx.design.OvsdLpPuFb2dTypeICostEvaluatorTestCase
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            isPext = false;
            gradExpctd = OvsdLpPuFb2dTypeICostEvaluatorTestCase.gradient(...
                lppufb,srcImgs{1}(:),coefs,scales,isPext)/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            [costActual,gradActual] = ...
                step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            %
            testCase.verifySize(gradActual,[numel(angs) 1]);
            diff = max(abs(gradExpctd(:)-gradActual(:)));
            testCase.verifyEqual(gradExpctd,gradActual,'AbsTol',1e-3,sprintf('%g',diff));            
            
        end
                        
        % Test
        function testAprxErrDec22Ch44Ord44Vm0Lv1GradObjOn(testCase)
            
            % Parameters
            width  = 16;
            height = 12;
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nVm = 0;
            nLevels = 1;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = ones(6,8);
            subCoefs{2}  = zeros(6,8);
            subCoefs{3}  = zeros(6,8);
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
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis2dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels...
                );
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'GradObj','on',...
                'IsFixedCoefs',true);
            
            % Expected values
            import saivdr.testcase.dictionary.nsoltx.design.OvsdLpPuFb2dTypeICostEvaluatorTestCase
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            isPext = false;
            gradExpctd = OvsdLpPuFb2dTypeICostEvaluatorTestCase.gradient(...
                lppufb,srcImgs{1}(:),coefs,scales,isPext)/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            [costActual,gradActual] = ...
                step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            %
            testCase.verifySize(gradActual,[numel(angs) 1]);
            diff = max(abs(gradExpctd(:)-gradActual(:)));
            testCase.verifyEqual(gradExpctd,gradActual,'AbsTol',1e-3,sprintf('%g',diff));            
            
        end
        
        % Test 
        function testAprxErrDec22Ch44Ord44Vm1Lv1PeriodicExtGradObjOn(testCase)
            
            % Parameters
            width  = 16;
            height = 12;
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 1;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = ones(6,8);
            subCoefs{2}  = zeros(6,8);
            subCoefs{3}  = zeros(6,8);
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
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis2dSystem(lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels...
                );
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Circular');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'GradObj','on',...
                'BoundaryOperation','Circular',...                
                'IsFixedCoefs',true);
            
            % Expected values
            import saivdr.testcase.dictionary.nsoltx.design.OvsdLpPuFb2dTypeICostEvaluatorTestCase
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            isPext = true;
            gradExpctd = OvsdLpPuFb2dTypeICostEvaluatorTestCase.gradient(...
                lppufb,srcImgs{1}(:),coefs,scales,isPext)/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            [costActual,gradActual] = ...
                step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            %
            testCase.verifySize(gradActual,[numel(angs) 1]);
            diff = max(abs(gradExpctd(:)-gradActual(:)));
            testCase.verifyEqual(gradExpctd,gradActual,'AbsTol',1e-3,sprintf('%g',diff));            
            
        end
                        
        % Test 
        function testAprxErrDec22Ch44Ord44Vm0Lv1PeriodicExtGradObjOn(testCase)
            
            % Parameters
            width  = 16;
            height = 12;
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nVm = 0;
            nLevels = 1;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = ones(6,8);
            subCoefs{2}  = zeros(6,8);
            subCoefs{3}  = zeros(6,8);
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
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis2dSystem(lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels...
                );
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Circular');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'GradObj','on',...
                'BoundaryOperation','Circular',...
                'IsFixedCoefs',true);
            
            % Expected values
            import saivdr.testcase.dictionary.nsoltx.design.OvsdLpPuFb2dTypeICostEvaluatorTestCase
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            isPext = true;
            gradExpctd = OvsdLpPuFb2dTypeICostEvaluatorTestCase.gradient(...
                lppufb,srcImgs{1}(:),coefs,scales,isPext)/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            [costActual,gradActual] = ...
                step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            %
            testCase.verifySize(gradActual,[numel(angs) 1]);
            diff = max(abs(gradExpctd(:)-gradActual(:)));
            testCase.verifyEqual(gradExpctd,gradActual,'AbsTol',1e-3,sprintf('%g',diff));
            
        end
        
        % Test
        function testAprxErrDec22Ch52Ord44Lv2(testCase)
            
            % Parameters
            width  = 16;
            height = 16;
            nDecs = [ 2 2 ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 2;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = ones(4,4);
            subCoefs{2}  = zeros(4,4);
            subCoefs{3}  = zeros(4,4);
            subCoefs{4}  = zeros(4,4);
            subCoefs{5}  = zeros(4,4);
            subCoefs{6}  = zeros(4,4);
            subCoefs{7}  = zeros(4,4);
            subCoefs{8}  = zeros(8,8);
            subCoefs{9}  = zeros(8,8);
            subCoefs{10} = zeros(8,8);
            subCoefs{11} = zeros(8,8);
            subCoefs{12} = zeros(8,8);
            subCoefs{13} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis2dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels...               
                );
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false);
            
            % Expected values
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test 
        function testAprxErrDec22Ch52Ord44Lv2PeriodicExt(testCase)
            
            % Parameters
            width  = 16;
            height = 16;
            nDecs = [ 2 2 ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 2;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = ones(4,4);
            subCoefs{2}  = zeros(4,4);
            subCoefs{3}  = zeros(4,4);
            subCoefs{4}  = zeros(4,4);
            subCoefs{5}  = zeros(4,4);
            subCoefs{6}  = zeros(4,4);
            subCoefs{7}  = zeros(4,4);
            subCoefs{8}  = zeros(8,8);
            subCoefs{9}  = zeros(8,8);
            subCoefs{10} = zeros(8,8);
            subCoefs{11} = zeros(8,8);
            subCoefs{12} = zeros(8,8);
            subCoefs{13} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis2dSystem(lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels...
                );
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Circular');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'BoundaryOperation','Circular',...
                'IsFixedCoefs',false);
            
            % Expected values
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
           
        % Test 
        function testAprxErrDec22Ch62Ord44Lv3(testCase)
            
            % Parameters
            width  = 16;
            height = 16;
            nDecs = [ 2 2 ];
            nChs  = [ 6 2 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 3;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = ones(2,2);
            subCoefs{2}  = zeros(2,2);
            subCoefs{3}  = zeros(2,2);
            subCoefs{4}  = zeros(2,2);
            subCoefs{5}  = zeros(2,2);
            subCoefs{6}  = zeros(2,2);
            subCoefs{7}  = zeros(2,2);
            subCoefs{8}  = zeros(2,2);
            subCoefs{9}  = zeros(4,4);
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
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis2dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false);
            
            % Expected values
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testAprxErrDec22Ch52Ord44Lv2MultiImgs(testCase)
            
            % Parameters
            width  = 16;
            height = 16;
            nDecs = [ 2 2 ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 2;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = -ones(4,4);
            subCoefs{2}  = zeros(4,4);
            subCoefs{3}  = zeros(4,4);
            subCoefs{4}  = zeros(4,4);
            subCoefs{5}  = zeros(4,4);
            subCoefs{6}  = zeros(4,4);
            subCoefs{7}  = zeros(4,4);
            subCoefs{8}  = zeros(8,8);
            subCoefs{9}  = zeros(8,8);
            subCoefs{10} = zeros(8,8);
            subCoefs{11} = zeros(8,8);
            subCoefs{12} = zeros(8,8);
            subCoefs{13} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales1 = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales1(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales1(iSubband,:))-1;
                mask1(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            %
            srcImgs{2} = rand(height,width);
            subCoefs{1}  = zeros(4,4);
            subCoefs{2}  = 2*ones(4,4);
            subCoefs{3}  = zeros(4,4);
            subCoefs{4}  = zeros(4,4);
            subCoefs{5}  = zeros(4,4);
            subCoefs{6}  = zeros(4,4);
            subCoefs{7}  = zeros(4,4);
            subCoefs{8}  = zeros(8,8);
            subCoefs{9}  = zeros(8,8);
            subCoefs{10} = zeros(8,8);
            subCoefs{11} = zeros(8,8);
            subCoefs{12} = zeros(8,8);
            subCoefs{13} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales2 = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales2(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales2(iSubband,:))-1;
                mask2(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer    = NsoltFactory.createAnalysis2dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false);
            
            % Expected values
            [coefs1,scales1]   = step(analyzer,srcImgs{1});
            coefs1 = coefs1.*mask1;
            aprxImg = step(synthesizer,coefs1,scales1);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            %
            [coefs2,scales2]   = step(analyzer,srcImgs{2});
            coefs2 = coefs2.*mask2;
            aprxImg = step(synthesizer,coefs2,scales2);
            costExpctd = costExpctd + norm(srcImgs{2}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs1;
            setOfScales{1} = scales1;
            sparseCoefs{2} = coefs2;
            setOfScales{2} = scales2;
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testSetLpPuFb2d(testCase)
            
            % Parameters
            width  = 16;
            height = 16;
            nDecs = [ 2 2 ];
            nChs  = [ 6 2 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 3;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = ones(2,2);
            subCoefs{2}  = zeros(2,2);
            subCoefs{3}  = zeros(2,2);
            subCoefs{4}  = zeros(2,2);
            subCoefs{5}  = zeros(2,2);
            subCoefs{6}  = zeros(2,2);
            subCoefs{7}  = zeros(2,2);
            subCoefs{8}  = zeros(2,2);
            subCoefs{9}  = zeros(4,4);
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
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false);
            
            % Pre
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', nVm,...
                'OutputMode','ParameterMatrixSet');
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costPre = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            % Pst
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            costPst = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(costPst(:) - costPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
            
        end
        
        % Test 
        function testIsFixedCoefsFalse(testCase)
            
            % Parameters
            width  = 16;
            height = 16;
            nDecs = [ 2 2 ];
            nChs  = [ 6 2 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 3;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = ones(2,2);
            subCoefs{2}  = ones(2,2);
            subCoefs{3}  = zeros(2,2);
            subCoefs{4}  = zeros(2,2);
            subCoefs{5}  = zeros(2,2);
            subCoefs{6}  = zeros(2,2);
            subCoefs{7}  = zeros(2,2);
            subCoefs{8}  = zeros(2,2);
            subCoefs{9}  = zeros(4,4);
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
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',true);
            
            % Pre
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds, ...
                'NumberOfVanishingMoments', nVm,...
                'OutputMode','ParameterMatrixSet');
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costPre = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            % Pst
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            costPst = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = abs(costPst - costPre);
            testCase.verifyThat(diff,IsGreaterThan(0));
            
        end
        
        % Test 
        function testAprxErrDec22Ch52Ord44Lv2MultiImgsFixedCoefs(testCase)
            
            % Parameters
            width  = 16;
            height = 16;
            nDecs = [ 2 2 ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 2;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = -ones(4,4);
            subCoefs{2}  = zeros(4,4);
            subCoefs{3}  = zeros(4,4);
            subCoefs{4}  = zeros(4,4);
            subCoefs{5}  = zeros(4,4);
            subCoefs{6}  = zeros(4,4);
            subCoefs{7}  = zeros(4,4);
            subCoefs{8}  = zeros(8,8);
            subCoefs{9}  = zeros(8,8);
            subCoefs{10} = zeros(8,8);
            subCoefs{11} = zeros(8,8);
            subCoefs{12} = zeros(8,8);
            subCoefs{13} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales1 = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales1(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales1(iSubband,:))-1;
                coefs1(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            %
            srcImgs{2} = rand(height,width);
            subCoefs{1}  = zeros(4,4);
            subCoefs{2}  = 2*ones(4,4);
            subCoefs{3}  = zeros(4,4);
            subCoefs{4}  = zeros(4,4);
            subCoefs{5}  = zeros(4,4);
            subCoefs{6}  = zeros(4,4);
            subCoefs{7}  = zeros(4,4);
            subCoefs{8}  = zeros(8,8);
            subCoefs{9}  = zeros(8,8);
            subCoefs{10} = zeros(8,8);
            subCoefs{11} = zeros(8,8);
            subCoefs{12} = zeros(8,8);
            subCoefs{13} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales2 = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales2(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales2(iSubband,:))-1;
                coefs2(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',true);
            
            % Expected values
            sparseCoefs{1} = coefs1;
            setOfScales{1} = scales1;
            aprxImg = step(synthesizer,sparseCoefs{1},setOfScales{1});
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            %
            sparseCoefs{2} = coefs2;
            setOfScales{2} = scales2;
            aprxImg = step(synthesizer,sparseCoefs{2},setOfScales{2});
            costExpctd = costExpctd + norm(srcImgs{2}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            
            % Actual values
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testClone(testCase)
            
            % Parameters
            width  = 16;
            height = 16;
            nDecs = [ 2 2 ];
            nChs  = [ 6 2 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 3;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = ones(2,2);
            subCoefs{2}  = zeros(2,2);
            subCoefs{3}  = zeros(2,2);
            subCoefs{4}  = zeros(2,2);
            subCoefs{5}  = zeros(2,2);
            subCoefs{6}  = zeros(2,2);
            subCoefs{7}  = zeros(2,2);
            subCoefs{8}  = zeros(2,2);
            subCoefs{9}  = zeros(4,4);
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
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false);
            
            % Pre
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', nVm,...
                'OutputMode','ParameterMatrixSet');
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costPre = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            % Clone
            aprxerrClone = clone(testCase.aprxerr);
            costCln = step(aprxerrClone,lppufb,sparseCoefs,setOfScales);
            
            % Evaluation
            testCase.verifyEqual(costPre,costCln,'RelTol',10e-10);
            
        end
        
        % Test 
        function testAprxErrDec222Ch66Ord222Lv2(testCase)
            
            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 6 ];
            nOrds = [ 2 2 2 ];
            nVm = 1;
            nLevels = 2;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1}  = ones(2,4,8);
            subCoefs{2}  = zeros(2,4,8);
            subCoefs{3}  = zeros(2,4,8);
            subCoefs{4}  = zeros(2,4,8);
            subCoefs{5}  = zeros(2,4,8);
            subCoefs{6}  = zeros(2,4,8);
            subCoefs{7}  = zeros(2,4,8);
            subCoefs{8}  = zeros(2,4,8);
            subCoefs{9}  = zeros(2,4,8);
            subCoefs{10} = zeros(2,4,8);
            subCoefs{11} = zeros(2,4,8);
            subCoefs{12} = zeros(2,4,8);
            subCoefs{13} = zeros(4,8,16);
            subCoefs{14} = zeros(4,8,16);
            subCoefs{15} = zeros(4,8,16);
            subCoefs{16} = zeros(4,8,16);
            subCoefs{17} = zeros(4,8,16);
            subCoefs{18} = zeros(4,8,16);
            subCoefs{19} = zeros(4,8,16);
            subCoefs{20} = zeros(4,8,16);
            subCoefs{21} = zeros(4,8,16);
            subCoefs{22} = zeros(4,8,16);
            subCoefs{23} = zeros(4,8,16);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis3dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels...
                );
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false);
            
            % Expected values
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test 
        function testAprxErrDec222Ch63Ord222Lv3(testCase)
            
            % Parameters
            height = 16;
            width  = 32;
            depth  = 64;
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 4 ];
            nOrds = [ 2 2 2 ];
            nVm = 1;
            nLevels = 3;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1}  = ones(2,4,8);
            subCoefs{2}  = zeros(2,4,8);
            subCoefs{3}  = zeros(2,4,8);
            subCoefs{4}  = zeros(2,4,8);
            subCoefs{5}  = zeros(2,4,8);
            subCoefs{6}  = zeros(2,4,8);
            subCoefs{7}  = zeros(2,4,8);
            subCoefs{8}  = zeros(2,4,8);
            subCoefs{9}  = zeros(2,4,8);
            subCoefs{10} = zeros(2,4,8);
            subCoefs{11} = zeros(4,8,16);
            subCoefs{12} = zeros(4,8,16);
            subCoefs{13} = zeros(4,8,16);
            subCoefs{14} = zeros(4,8,16);
            subCoefs{15} = zeros(4,8,16);
            subCoefs{16} = zeros(4,8,16);
            subCoefs{17} = zeros(4,8,16);
            subCoefs{18} = zeros(4,8,16);
            subCoefs{19} = zeros(4,8,16);
            subCoefs{20} = zeros(8,16,32);
            subCoefs{21} = zeros(8,16,32);
            subCoefs{22} = zeros(8,16,32);
            subCoefs{23} = zeros(8,16,32);
            subCoefs{24} = zeros(8,16,32);
            subCoefs{25} = zeros(8,16,32);
            subCoefs{26} = zeros(8,16,32);
            subCoefs{27} = zeros(8,16,32);
            subCoefs{28} = zeros(8,16,32);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis3dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false);
            
            % Expected values
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testAprxErrDec222Ch66Ord444Lv2MultiImgs(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 5 ];
            nOrds = [ 4 4 4 ];
            nVm = 1;
            nLevels = 2;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1}  = -ones(4,4,4);
            subCoefs{2}  = zeros(4,4,4);
            subCoefs{3}  = zeros(4,4,4);
            subCoefs{4}  = zeros(4,4,4);
            subCoefs{5}  = zeros(4,4,4);
            subCoefs{6}  = zeros(4,4,4);
            subCoefs{7}  = zeros(4,4,4);
            subCoefs{8}  = zeros(4,4,4);
            subCoefs{9}  = zeros(4,4,4);
            subCoefs{10} = zeros(4,4,4);
            subCoefs{11} = zeros(8,8,8);
            subCoefs{12} = zeros(8,8,8);
            subCoefs{13} = zeros(8,8,8);
            subCoefs{14} = zeros(8,8,8);
            subCoefs{15} = zeros(8,8,8);
            subCoefs{16} = zeros(8,8,8);
            subCoefs{17} = zeros(8,8,8);
            subCoefs{18} = zeros(8,8,8);
            subCoefs{19} = zeros(8,8,8);
            nSubbands = length(subCoefs);
            scales1 = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales1(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales1(iSubband,:))-1;
                mask1(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            %
            srcImgs{2} = rand(height,width,depth);
            subCoefs{1}  = zeros(4,4,4);
            subCoefs{2}  = 2*ones(4,4,4);
            subCoefs{3}  = zeros(4,4,4);
            subCoefs{4}  = zeros(4,4,4);
            subCoefs{5}  = zeros(4,4,4);
            subCoefs{6}  = zeros(4,4,4);
            subCoefs{7}  = zeros(4,4,4);
            subCoefs{8}  = zeros(4,4,4);
            subCoefs{9}  = zeros(4,4,4);
            subCoefs{10} = zeros(4,4,4);
            subCoefs{11} = zeros(8,8,8);
            subCoefs{12} = zeros(8,8,8);
            subCoefs{13} = zeros(8,8,8);
            subCoefs{14} = zeros(8,8,8);
            subCoefs{15} = zeros(8,8,8);
            subCoefs{16} = zeros(8,8,8);
            subCoefs{17} = zeros(8,8,8);
            subCoefs{18} = zeros(8,8,8);
            subCoefs{19} = zeros(8,8,8);
            nSubbands = length(subCoefs);
            scales2 = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales2(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales2(iSubband,:))-1;
                mask2(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer    = NsoltFactory.createAnalysis3dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false);
            
            % Expected values
            [coefs1,scales1]   = step(analyzer,srcImgs{1});
            coefs1 = coefs1.*mask1;
            aprxImg = step(synthesizer,coefs1,scales1);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            %
            [coefs2,scales2]   = step(analyzer,srcImgs{2});
            coefs2 = coefs2.*mask2;
            aprxImg = step(synthesizer,coefs2,scales2);
            costExpctd = costExpctd + norm(srcImgs{2}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs1;
            setOfScales{1} = scales1;
            sparseCoefs{2} = coefs2;
            setOfScales{2} = scales2;
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test 
        function testSetLpPuFb3d(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 2 ];
            nVm = 1;
            nLevels = 3;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1}  = ones(2,2,2);
            subCoefs{2}  = zeros(2,2,2);
            subCoefs{3}  = zeros(2,2,2);
            subCoefs{4}  = zeros(2,2,2);
            subCoefs{5}  = zeros(2,2,2);
            subCoefs{6}  = zeros(2,2,2);
            subCoefs{7}  = zeros(2,2,2);
            subCoefs{8}  = zeros(2,2,2);
            subCoefs{9}  = zeros(2,2,2);
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
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false);
            
            % Pre
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', nVm,...
                'OutputMode','ParameterMatrixSet');
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costPre = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            % Pst
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            costPst = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(costPst(:) - costPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
            
        end
        
        % Test 
        function testIsFixedCoefsFalse3d(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 4 ];
            nOrds = [ 2 2 2 ];
            nVm = 1;
            nLevels = 3;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1}  = ones(2,2,2);
            subCoefs{2}  = ones(2,2,2);
            subCoefs{3}  = ones(2,2,2);
            subCoefs{4}  = ones(2,2,2);
            subCoefs{5}  = zeros(2,2,2);
            subCoefs{6}  = zeros(2,2,2);
            subCoefs{7}  = zeros(2,2,2);
            subCoefs{8}  = zeros(2,2,2);
            subCoefs{9}  = zeros(2,2,2);
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
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',true);
            
            % Pre
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds, ...
                'NumberOfVanishingMoments', nVm,...
                'OutputMode','ParameterMatrixSet');
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costPre = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            % Pst
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            mus = get(lppufb,'Mus');
            mus = 1-2*(rand(size(mus))<0.5);
            set(lppufb,'Mus',mus);            
            costPst = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = abs(costPst - costPre);
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
            
        end
              
        % Test 
        function testAprxErrDec222Ch54Ord444Lv2MultiImgsFixedCoefs(testCase)
            
            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 4 4 4 ];
            nVm = 1;
            nLevels = 2;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1}  = -ones(2,4,8);
            subCoefs{2}  = zeros(2,4,8);
            subCoefs{3}  = zeros(2,4,8);
            subCoefs{4}  = zeros(2,4,8);
            subCoefs{5}  = zeros(2,4,8);
            subCoefs{6}  = zeros(2,4,8);
            subCoefs{7}  = zeros(2,4,8);
            subCoefs{8}  = zeros(2,4,8);
            subCoefs{9}  = zeros(2,4,8);
            subCoefs{10} = zeros(4,8,16);
            subCoefs{11} = zeros(4,8,16);
            subCoefs{12} = zeros(4,8,16);
            subCoefs{13} = zeros(4,8,16);
            subCoefs{14} = zeros(4,8,16);
            subCoefs{15} = zeros(4,8,16);
            subCoefs{16} = zeros(4,8,16);
            subCoefs{17} = zeros(4,8,16);
            nSubbands = length(subCoefs);
            scales1 = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales1(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales1(iSubband,:))-1;
                coefs1(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            %
            srcImgs{2} = rand(height,width,depth);
            subCoefs{1}  = zeros(2,4,8);
            subCoefs{2}  = 2*ones(2,4,8);
            subCoefs{3}  = zeros(2,4,8);
            subCoefs{4}  = zeros(2,4,8);
            subCoefs{5}  = zeros(2,4,8);
            subCoefs{6}  = zeros(2,4,8);
            subCoefs{7}  = zeros(2,4,8);
            subCoefs{8}  = zeros(2,4,8);
            subCoefs{9}  = zeros(2,4,8);
            subCoefs{10} = zeros(4,8,16);
            subCoefs{11} = zeros(4,8,16);
            subCoefs{12} = zeros(4,8,16);
            subCoefs{13} = zeros(4,8,16);
            subCoefs{14} = zeros(4,8,16);
            subCoefs{15} = zeros(4,8,16);
            subCoefs{16} = zeros(4,8,16);
            subCoefs{17} = zeros(4,8,16);
            nSubbands = length(subCoefs);
            scales2 = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales2(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales2(iSubband,:))-1;
                coefs2(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',true);
            
            % Expected values
            sparseCoefs{1} = coefs1;
            setOfScales{1} = scales1;
            aprxImg = step(synthesizer,sparseCoefs{1},setOfScales{1});
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            %
            sparseCoefs{2} = coefs2;
            setOfScales{2} = scales2;
            aprxImg = step(synthesizer,sparseCoefs{2},setOfScales{2});
            costExpctd = costExpctd + norm(srcImgs{2}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            
            % Actual values
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test
        function testClone3d(testCase)
            
            % Parameters
            width  = 16;
            height = 16;
            depth  = 16;
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 4 ];
            nOrds = [ 2 2 2 ];
            nVm = 1;
            nLevels = 3;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1}  = ones(2,2,2);
            subCoefs{2}  = zeros(2,2,2);
            subCoefs{3}  = zeros(2,2,2);
            subCoefs{4}  = zeros(2,2,2);
            subCoefs{5}  = zeros(2,2,2);
            subCoefs{6}  = zeros(2,2,2);
            subCoefs{7}  = zeros(2,2,2);
            subCoefs{8}  = zeros(2,2,2);
            subCoefs{9}  = zeros(2,2,2);
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
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false);
            
            % Pre
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', nVm,...
                'OutputMode','ParameterMatrixSet');
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costPre = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            % Clone
            aprxerrClone = clone(testCase.aprxerr);
            costCln = step(aprxerrClone,lppufb,sparseCoefs,setOfScales);
            
            % Evaluation
            testCase.verifyEqual(costPre,costCln,'RelTol',10e-10);
            
        end
        
        % Test
        function testAprxErrDec22Ch44Ord44Lv1Sgd(testCase)
            
            % Parameters
            width  = 12;
            height = 16;
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 1;
            srcImgs{1} = rand(height,width);
            subCoefs{1} = -ones(6,8);
            subCoefs{2} = zeros(6,8);
            subCoefs{3} = zeros(6,8);
            subCoefs{4} = zeros(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = zeros(6,8);
            subCoefs{8} = zeros(6,8);
            nSubbands = length(subCoefs);
            scales1 = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales1(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales1(iSubband,:))-1;
                mask1(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            %
            srcImgs{2} = rand(height,width);
            subCoefs{1}  = zeros(6,8);
            subCoefs{2}  = 2*ones(6,8);
            subCoefs{3}  = zeros(6,8);
            subCoefs{4}  = zeros(6,8);
            subCoefs{5}  = zeros(6,8);
            subCoefs{6}  = zeros(6,8);
            subCoefs{7}  = zeros(6,8);
            subCoefs{8}  = zeros(6,8);
            nSubbands = length(subCoefs);
            scales2 = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales2(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales2(iSubband,:))-1;
                mask2(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer    = NsoltFactory.createAnalysis2dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false,...
                'Stochastic','on');
            
            % Expected values
            [coefs1,scales1]   = step(analyzer,srcImgs{1});
            coefs1 = coefs1.*mask1;
            aprxImg1 = step(synthesizer,coefs1,scales1);
            costExpctd1 = sum((srcImgs{1}(:) - aprxImg1(:)).^2)/numel(srcImgs{1});
            %
            [coefs2,scales2]   = step(analyzer,srcImgs{2});
            coefs2 = coefs2.*mask2;
            aprxImg2 = step(synthesizer,coefs2,scales2);
            costExpctd2 = sum((srcImgs{2}(:) - aprxImg2(:)).^2)/numel(srcImgs{2});
            %
            costExpctd = (costExpctd1+costExpctd2)/2;
            
            % Actual values
            sparseCoefs{1} = coefs1;
            setOfScales{1} = scales1;
            sparseCoefs{2} = coefs2;
            setOfScales{2} = scales2;
            costActual1 = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales,1);
            costActual2 = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales,2);
            %
            costActual  = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales,[]);
            
            %
            diff = norm(costExpctd1-costActual1)/norm(costExpctd1);
            testCase.verifyEqual(costExpctd1,costActual1,'RelTol',1e-10,sprintf('%g',diff));
            diff = norm(costExpctd2-costActual2)/norm(costExpctd2);
            testCase.verifyEqual(costExpctd2,costActual2,'RelTol',1e-10,sprintf('%g',diff));            
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));                        
            
        end
        
        % Test 
        function testAprxErrDec22Ch44Ord44Lv1SgdMseOff(testCase)
            
            % Parameters
            width  = 12;
            height = 16;
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 1;
            srcImgs{1} = rand(height,width);
            subCoefs{1} = -ones(6,8);
            subCoefs{2} = zeros(6,8);
            subCoefs{3} = zeros(6,8);
            subCoefs{4} = zeros(6,8);
            subCoefs{5} = zeros(6,8);
            subCoefs{6} = zeros(6,8);
            subCoefs{7} = zeros(6,8);
            subCoefs{8} = zeros(6,8);
            nSubbands = length(subCoefs);
            scales1 = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales1(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales1(iSubband,:))-1;
                mask1(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            %
            srcImgs{2} = rand(height,width);
            subCoefs{1}  = zeros(6,8);
            subCoefs{2}  = 2*ones(6,8);
            subCoefs{3}  = zeros(6,8);
            subCoefs{4}  = zeros(6,8);
            subCoefs{5}  = zeros(6,8);
            subCoefs{6}  = zeros(6,8);
            subCoefs{7}  = zeros(6,8);
            subCoefs{8}  = zeros(6,8);
            nSubbands = length(subCoefs);
            scales2 = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales2(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales2(iSubband,:))-1;
                mask2(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer    = NsoltFactory.createAnalysis2dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false,...
                'Mse','off',...
                'Stochastic','on');
            
            % Expected values
            [coefs1,scales1]   = step(analyzer,srcImgs{1});
            coefs1 = coefs1.*mask1;
            aprxImg1 = step(synthesizer,coefs1,scales1);
            costExpctd1 = sum((srcImgs{1}(:) - aprxImg1(:)).^2);
            %
            [coefs2,scales2]   = step(analyzer,srcImgs{2});
            coefs2 = coefs2.*mask2;
            aprxImg2 = step(synthesizer,coefs2,scales2);
            costExpctd2 = sum((srcImgs{2}(:) - aprxImg2(:)).^2);
            %
            costExpctd = (costExpctd1+costExpctd2)/2;
            
            % Actual values
            sparseCoefs{1} = coefs1;
            setOfScales{1} = scales1;
            sparseCoefs{2} = coefs2;
            setOfScales{2} = scales2;
            costActual1 = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales,1);
            costActual2 = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales,2);
            %
            costActual  = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales,[]);
            
            %
            diff = norm(costExpctd1-costActual1)/norm(costExpctd1);
            testCase.verifyEqual(costExpctd1,costActual1,'RelTol',1e-10,sprintf('%g',diff));
            diff = norm(costExpctd2-costActual2)/norm(costExpctd2);
            testCase.verifyEqual(costExpctd2,costActual2,'RelTol',1e-10,sprintf('%g',diff));            
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));                        
            
        end
        
        % Test 
        function testAprxErrDec22Ch44Ord44Vm1Lv1GradObjOnMseOff(testCase)
            
            % Parameters
            width  = 16;
            height = 12;
            nDecs = [ 2 2 ];
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 1;
            srcImgs{1} = rand(height,width);
            subCoefs{1} = ones(6,8);
            subCoefs{2} = zeros(6,8);
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
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis2dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels...
                );
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'GradObj','on',...
                'Mse','off',...
                'IsFixedCoefs',true);
            
            % Expected values
            import saivdr.testcase.dictionary.nsoltx.design.OvsdLpPuFb2dTypeICostEvaluatorTestCase
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/length(srcImgs);
            isPext = false;
            gradExpctd = OvsdLpPuFb2dTypeICostEvaluatorTestCase.gradient(...
                lppufb,srcImgs{1}(:),coefs,scales,isPext)/length(srcImgs);
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            [costActual,gradActual] = ...
                step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            %
            testCase.verifySize(gradActual,[numel(angs) 1]);
            diff = max(abs(gradExpctd(:)-gradActual(:)));
            testCase.verifyEqual(gradExpctd,gradActual,'AbsTol',1e-3,sprintf('%g',diff));            
            
        end
               
        % Test 
        function testAprxErrDec22Ch52Ord44Lv2MultiImgsMseOff(testCase)
            
            % Parameters
            width  = 16;
            height = 16;
            nDecs = [ 2 2 ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 2;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = -ones(4,4);
            subCoefs{2}  = zeros(4,4);
            subCoefs{3}  = zeros(4,4);
            subCoefs{4}  = zeros(4,4);
            subCoefs{5}  = zeros(4,4);
            subCoefs{6}  = zeros(4,4);
            subCoefs{7}  = zeros(4,4);
            subCoefs{8}  = zeros(8,8);
            subCoefs{9}  = zeros(8,8);
            subCoefs{10} = zeros(8,8);
            subCoefs{11} = zeros(8,8);
            subCoefs{12} = zeros(8,8);
            subCoefs{13} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales1 = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales1(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales1(iSubband,:))-1;
                mask1(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            %
            srcImgs{2} = rand(height,width);
            subCoefs{1}  = zeros(4,4);
            subCoefs{2}  = 2*ones(4,4);
            subCoefs{3}  = zeros(4,4);
            subCoefs{4}  = zeros(4,4);
            subCoefs{5}  = zeros(4,4);
            subCoefs{6}  = zeros(4,4);
            subCoefs{7}  = zeros(4,4);
            subCoefs{8}  = zeros(8,8);
            subCoefs{9}  = zeros(8,8);
            subCoefs{10} = zeros(8,8);
            subCoefs{11} = zeros(8,8);
            subCoefs{12} = zeros(8,8);
            subCoefs{13} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales2 = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales2(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales2(iSubband,:))-1;
                mask2(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer    = NsoltFactory.createAnalysis2dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'Mse','off',...
                'IsFixedCoefs',false);
            
            % Expected values
            [coefs1,scales1]   = step(analyzer,srcImgs{1});
            coefs1 = coefs1.*mask1;
            aprxImg = step(synthesizer,coefs1,scales1);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            %
            [coefs2,scales2]   = step(analyzer,srcImgs{2});
            coefs2 = coefs2.*mask2;
            aprxImg = step(synthesizer,coefs2,scales2);
            costExpctd = costExpctd + norm(srcImgs{2}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/length(srcImgs);
            
            % Actual values
            sparseCoefs{1} = coefs1;
            setOfScales{1} = scales1;
            sparseCoefs{2} = coefs2;
            setOfScales{2} = scales2;
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
                 
        % Test
        function testAprxErrDec22Ch52Ord44Lv2MultiImgsFixedCoefsMseOff(testCase)
            
            % Parameters
            width  = 16;
            height = 16;
            nDecs = [ 2 2 ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            nVm = 1;
            nLevels = 2;
            srcImgs{1} = rand(height,width);
            subCoefs{1}  = -ones(4,4);
            subCoefs{2}  = zeros(4,4);
            subCoefs{3}  = zeros(4,4);
            subCoefs{4}  = zeros(4,4);
            subCoefs{5}  = zeros(4,4);
            subCoefs{6}  = zeros(4,4);
            subCoefs{7}  = zeros(4,4);
            subCoefs{8}  = zeros(8,8);
            subCoefs{9}  = zeros(8,8);
            subCoefs{10} = zeros(8,8);
            subCoefs{11} = zeros(8,8);
            subCoefs{12} = zeros(8,8);
            subCoefs{13} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales1 = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales1(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales1(iSubband,:))-1;
                coefs1(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            %
            srcImgs{2} = rand(height,width);
            subCoefs{1}  = zeros(4,4);
            subCoefs{2}  = 2*ones(4,4);
            subCoefs{3}  = zeros(4,4);
            subCoefs{4}  = zeros(4,4);
            subCoefs{5}  = zeros(4,4);
            subCoefs{6}  = zeros(4,4);
            subCoefs{7}  = zeros(4,4);
            subCoefs{8}  = zeros(8,8);
            subCoefs{9}  = zeros(8,8);
            subCoefs{10} = zeros(8,8);
            subCoefs{11} = zeros(8,8);
            subCoefs{12} = zeros(8,8);
            subCoefs{13} = zeros(8,8);
            nSubbands = length(subCoefs);
            scales2 = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales2(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales2(iSubband,:))-1;
                coefs2(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'Mse','off',...
                'IsFixedCoefs',true);
            
            % Expected values
            sparseCoefs{1} = coefs1;
            setOfScales{1} = scales1;
            aprxImg = step(synthesizer,sparseCoefs{1},setOfScales{1});
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            %
            sparseCoefs{2} = coefs2;
            setOfScales{2} = scales2;
            aprxImg = step(synthesizer,sparseCoefs{2},setOfScales{2});
            costExpctd = costExpctd + norm(srcImgs{2}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/length(srcImgs);
            
            % Actual values
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
        
        % Test 
        function testAprxErrDec222Ch55Ord222Vm0Lv1GradObjOn(testCase)
            
            % Parameters
            width  = 16;
            height = 12;
            depth  = 20;
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 2 ];
            nVm = 0;
            nLevels = 1;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1}  = ones(6,8,10);
            subCoefs{2}  = zeros(6,8,10);
            subCoefs{3}  = zeros(6,8,10);
            subCoefs{4} = zeros(6,8,10);
            subCoefs{5} = zeros(6,8,10);
            subCoefs{6} = zeros(6,8,10);
            subCoefs{7} = zeros(6,8,10);
            subCoefs{8} = zeros(6,8,10);
            subCoefs{9} = zeros(6,8,10);
            subCoefs{10} = zeros(6,8,10);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis3dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels...
                );
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'GradObj','on',...
                'IsFixedCoefs',true);
            
            % Expected values
            import saivdr.testcase.dictionary.nsoltx.design.OvsdLpPuFb3dTypeICostEvaluatorTestCase
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            isPext = false;
            gradExpctd = OvsdLpPuFb3dTypeICostEvaluatorTestCase.gradient(...
                lppufb,srcImgs{1}(:),coefs,scales,isPext)/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            [costActual,gradActual] = ...
                step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            %
            testCase.verifySize(gradActual,[numel(angs) 1]);
            diff = max(abs(gradExpctd(:)-gradActual(:)));
            testCase.verifyEqual(gradExpctd,gradActual,'AbsTol',1e-3,sprintf('%g',diff));            
            
        end
        
        % Test 
        function testAprxErrDec222Ch55Ord222Vm1Lv1PeriodicExtGradObjOn(testCase)
            
            % Parameters
            width  = 16;
            height = 12;
            depth  = 20;
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 2 ];
            nVm = 1;
            nLevels = 1;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1}  = ones(6,8,10);
            subCoefs{2}  = zeros(6,8,10);
            subCoefs{3}  = zeros(6,8,10);
            subCoefs{4} = zeros(6,8,10);
            subCoefs{5} = zeros(6,8,10);
            subCoefs{6} = zeros(6,8,10);
            subCoefs{7} = zeros(6,8,10);
            subCoefs{8} = zeros(6,8,10);
            subCoefs{9} = zeros(6,8,10);
            subCoefs{10} = zeros(6,8,10);
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis3dSystem(lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels...                
                );
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufb,...
                'BoundaryOperation','Circular');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'GradObj','on',...
                'BoundaryOperation','Circular',...                
                'IsFixedCoefs',true);
            
            % Expected values
            import saivdr.testcase.dictionary.nsoltx.design.OvsdLpPuFb3dTypeICostEvaluatorTestCase
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            isPext = true;
            gradExpctd = OvsdLpPuFb3dTypeICostEvaluatorTestCase.gradient(...
                lppufb,srcImgs{1}(:),coefs,scales,isPext)/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            [costActual,gradActual] = ...
                step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            %
            testCase.verifySize(gradActual,[numel(angs) 1]);
            diff = max(abs(gradExpctd(:)-gradActual(:)));
            testCase.verifyEqual(gradExpctd,gradActual,'AbsTol',1e-3,sprintf('%g',diff));            
            
        end

        % Test 
        function testAprxErrDec222Ch55Ord222Vm0Lv1PeriodicExtGradObjOn(testCase)
            
            % Parameters
            width  = 16;
            height = 12;
            depth  = 20;
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 2 ];
            nVm = 0;
            nLevels = 1;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1}  = ones(6,8,10);
            subCoefs{2}  = zeros(6,8,10);
            subCoefs{3}  = zeros(6,8,10);
            subCoefs{4} = zeros(6,8,10);
            subCoefs{5} = zeros(6,8,10);
            subCoefs{6} = zeros(6,8,10);
            subCoefs{7} = zeros(6,8,10);
            subCoefs{8} = zeros(6,8,10);
            subCoefs{9} = zeros(6,8,10);
            subCoefs{10} = zeros(6,8,10);            
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis3dSystem(lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels...
                );
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufb,...
                'BoundaryOperation','Circular');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'GradObj','on',...
                'BoundaryOperation','Circular',...
                'IsFixedCoefs',true);
            
            % Expected values
            import saivdr.testcase.dictionary.nsoltx.design.OvsdLpPuFb3dTypeICostEvaluatorTestCase
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/numel(cell2mat(srcImgs));
            isPext = true;
            gradExpctd = OvsdLpPuFb3dTypeICostEvaluatorTestCase.gradient(...
                lppufb,srcImgs{1}(:),coefs,scales,isPext)/numel(cell2mat(srcImgs));
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            [costActual,gradActual] = ...
                step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            %
            testCase.verifySize(gradActual,[numel(angs) 1]);
            diff = max(abs(gradExpctd(:)-gradActual(:)));
            testCase.verifyEqual(gradExpctd,gradActual,'AbsTol',1e-3,sprintf('%g',diff));
            
        end
        
        % Test
        function testAprxErrDec222Ch55Ord222Lv1Sgd(testCase)
            
            % Parameters
            width  = 12;
            height = 16;
            depth  = 20;
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 2 ];
            nVm = 1;
            nLevels = 1;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1} = -ones(6,8,10);
            subCoefs{2} = zeros(6,8,10);
            subCoefs{3} = zeros(6,8,10);
            subCoefs{4} = zeros(6,8,10);
            subCoefs{5} = zeros(6,8,10);
            subCoefs{6} = zeros(6,8,10);
            subCoefs{7} = zeros(6,8,10);
            subCoefs{8} = zeros(6,8,10);
            subCoefs{9} = zeros(6,8,10);
            subCoefs{10} = zeros(6,8,10);
            nSubbands = length(subCoefs);
            scales1 = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales1(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales1(iSubband,:))-1;
                mask1(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            %
            srcImgs{2} = rand(height,width,depth);
            subCoefs{1}  = zeros(6,8,10);
            subCoefs{2}  = 2*ones(6,8,10);
            subCoefs{3}  = zeros(6,8,10);
            subCoefs{4}  = zeros(6,8,10);
            subCoefs{5}  = zeros(6,8,10);
            subCoefs{6}  = zeros(6,8,10);
            subCoefs{7}  = zeros(6,8,10);
            subCoefs{8}  = zeros(6,8,10);
            subCoefs{9}  = zeros(6,8,10);
            subCoefs{10}  = zeros(6,8,10);            
            nSubbands = length(subCoefs);
            scales2 = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales2(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales2(iSubband,:))-1;
                mask2(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer    = NsoltFactory.createAnalysis3dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false,...
                'Stochastic','on');
            
            % Expected values
            [coefs1,scales1]   = step(analyzer,srcImgs{1});
            coefs1 = coefs1.*mask1;
            aprxImg1 = step(synthesizer,coefs1,scales1);
            costExpctd1 = sum((srcImgs{1}(:) - aprxImg1(:)).^2)/numel(srcImgs{1});
            %
            [coefs2,scales2]   = step(analyzer,srcImgs{2});
            coefs2 = coefs2.*mask2;
            aprxImg2 = step(synthesizer,coefs2,scales2);
            costExpctd2 = sum((srcImgs{2}(:) - aprxImg2(:)).^2)/numel(srcImgs{2});
            %
            costExpctd = (costExpctd1+costExpctd2)/2;
            
            % Actual values
            sparseCoefs{1} = coefs1;
            setOfScales{1} = scales1;
            sparseCoefs{2} = coefs2;
            setOfScales{2} = scales2;
            costActual1 = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales,1);
            costActual2 = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales,2);
            %
            costActual  = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales,[]);
            
            %
            diff = norm(costExpctd1-costActual1)/norm(costExpctd1);
            testCase.verifyEqual(costExpctd1,costActual1,'RelTol',1e-10,sprintf('%g',diff));
            diff = norm(costExpctd2-costActual2)/norm(costExpctd2);
            testCase.verifyEqual(costExpctd2,costActual2,'RelTol',1e-10,sprintf('%g',diff));            
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));                        
            
        end
        
        % Test
        function testAprxErrDec222Ch55Ord222Lv1SgdMseOff(testCase)
            
            % Parameters
            width  = 12;
            height = 16;
            depth  = 20;
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 2 ];
            nVm = 1;
            nLevels = 1;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1} = -ones(6,8,10);
            subCoefs{2} = zeros(6,8,10);
            subCoefs{3} = zeros(6,8,10);
            subCoefs{4} = zeros(6,8,10);
            subCoefs{5} = zeros(6,8,10);
            subCoefs{6} = zeros(6,8,10);
            subCoefs{7} = zeros(6,8,10);
            subCoefs{8} = zeros(6,8,10);
            subCoefs{9} = zeros(6,8,10);
            subCoefs{10} = zeros(6,8,10);            
            nSubbands = length(subCoefs);
            scales1 = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales1(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales1(iSubband,:))-1;
                mask1(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            %
            srcImgs{2} = rand(height,width,depth);
            subCoefs{1}  = zeros(6,8,10);
            subCoefs{2}  = 2*ones(6,8,10);
            subCoefs{3}  = zeros(6,8,10);
            subCoefs{4}  = zeros(6,8,10);
            subCoefs{5}  = zeros(6,8,10);
            subCoefs{6}  = zeros(6,8,10);
            subCoefs{7}  = zeros(6,8,10);
            subCoefs{8}  = zeros(6,8,10);
            subCoefs{9}  = zeros(6,8,10);
            subCoefs{10}  = zeros(6,8,10);            
            nSubbands = length(subCoefs);
            scales2 = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales2(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales2(iSubband,:))-1;
                mask2(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer    = NsoltFactory.createAnalysis3dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'IsFixedCoefs',false,...
                'Mse','off',...
                'Stochastic','on');
            
            % Expected values
            [coefs1,scales1]   = step(analyzer,srcImgs{1});
            coefs1 = coefs1.*mask1;
            aprxImg1 = step(synthesizer,coefs1,scales1);
            costExpctd1 = sum((srcImgs{1}(:) - aprxImg1(:)).^2);
            %
            [coefs2,scales2]   = step(analyzer,srcImgs{2});
            coefs2 = coefs2.*mask2;
            aprxImg2 = step(synthesizer,coefs2,scales2);
            costExpctd2 = sum((srcImgs{2}(:) - aprxImg2(:)).^2);
            %
            costExpctd = (costExpctd1+costExpctd2)/2;
            
            % Actual values
            sparseCoefs{1} = coefs1;
            setOfScales{1} = scales1;
            sparseCoefs{2} = coefs2;
            setOfScales{2} = scales2;
            costActual1 = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales,1);
            costActual2 = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales,2);
            %
            costActual  = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales,[]);
            
            %
            diff = norm(costExpctd1-costActual1)/norm(costExpctd1);
            testCase.verifyEqual(costExpctd1,costActual1,'RelTol',1e-10,sprintf('%g',diff));
            diff = norm(costExpctd2-costActual2)/norm(costExpctd2);
            testCase.verifyEqual(costExpctd2,costActual2,'RelTol',1e-10,sprintf('%g',diff));            
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));                        
            
        end
        
        % Test 
        function testAprxErrDec222Ch55Ord222Vm1Lv1GradObjOnMseOff(testCase)
            
            % Parameters
            width  = 16;
            height = 12;
            depth  = 20;
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 2 ];
            nVm = 1;
            nLevels = 1;
            srcImgs{1} = rand(height,width,depth);
            subCoefs{1} = ones(6,8,10);
            subCoefs{2} = zeros(6,8,10);
            subCoefs{3} = zeros(6,8,10);
            subCoefs{4} = zeros(6,8,10);
            subCoefs{5} = zeros(6,8,10);
            subCoefs{6} = zeros(6,8,10);
            subCoefs{7} = zeros(6,8,10);
            subCoefs{8} = zeros(6,8,10);
            subCoefs{9} = zeros(6,8,10);
            subCoefs{10} = zeros(6,8,10);            
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,3);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                masks(sIdx:eIdx) = (subCoefs{iSubband}(:).'~=0);
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsoltx.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysis3dSystem(lppufb,...
                'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels....
                );
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels,...
                'GradObj','on',...
                'Mse','off',...
                'IsFixedCoefs',true);
            
            % Expected values
            import saivdr.testcase.dictionary.nsoltx.design.OvsdLpPuFb3dTypeICostEvaluatorTestCase
            [coefs,scales] = step(analyzer,srcImgs{1});
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            costExpctd = costExpctd/length(srcImgs);
            isPext = false;
            gradExpctd = OvsdLpPuFb3dTypeICostEvaluatorTestCase.gradient(...
                lppufb,srcImgs{1}(:),coefs,scales,isPext)/length(srcImgs);
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            [costActual,gradActual] = ...
                step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            %
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            %
            testCase.verifySize(gradActual,[numel(angs) 1]);
            diff = max(abs(gradExpctd(:)-gradActual(:)));
            testCase.verifyEqual(gradExpctd,gradActual,'AbsTol',1e-3,sprintf('%g',diff));            
            
        end

    end
end

