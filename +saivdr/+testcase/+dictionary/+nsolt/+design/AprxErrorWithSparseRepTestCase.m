classdef AprxErrorWithSparseRepTestCase < matlab.unittest.TestCase
    %APRXERRORWITHSPARSEREPTESTCASE Test case for AprxErrorWithSparseRep
    %
    % SVN identifier:
    % $Id: AprxErrorWithSparseRepTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        aprxerr
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.aprxerr);
        end
    end
    
    methods (Test)
        
        % Test for default construction
        function testAprxErrDec22Ch5plus2Ord44Lv2(testCase)
            
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
            import saivdr.dictionary.nsolt.*
            import saivdr.dictionary.nsolt.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysisSystem(lppufb,...
                'BoundaryOperation','Termination'...
                );
            synthesizer = NsoltFactory.createSynthesisSystem(lppufb,...
                'BoundaryOperation','Termination');            
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels,...
                'IsFixedCoefs',false);
            
            % Expected values
            [coefs,scales] = step(analyzer,srcImgs{1},nLevels);
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            % 
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
  
        % Test for default construction
        function testAprxErrDec22Ch6plus2Ord44Lv3(testCase)
            
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
            import saivdr.dictionary.nsolt.*
            import saivdr.dictionary.nsolt.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer = NsoltFactory.createAnalysisSystem(lppufb,...
                'BoundaryOperation','Termination');            
            synthesizer = NsoltFactory.createSynthesisSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels,...
                'IsFixedCoefs',false);
            
            % Expected values
            [coefs,scales] = step(analyzer,srcImgs{1},nLevels);
            coefs = coefs.*masks;
            aprxImg = step(synthesizer,coefs,scales);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            
            % Actual values
            sparseCoefs{1} = coefs;
            setOfScales{1} = scales;
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);

            % 
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end
              
        % Test for default construction
        function testAprxErrDec22Ch5plus2Ord44Lv2MultiImgs(testCase)
            
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
            import saivdr.dictionary.nsolt.*
            import saivdr.dictionary.nsolt.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            analyzer    = NsoltFactory.createAnalysisSystem(lppufb,...
                'BoundaryOperation','Termination');
            synthesizer = NsoltFactory.createSynthesisSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels,...
                'IsFixedCoefs',false);

            % Expected values
            [coefs1,scales1]   = step(analyzer,srcImgs{1},nLevels);            
            coefs1 = coefs1.*mask1;
            aprxImg = step(synthesizer,coefs1,scales1);
            costExpctd = norm(srcImgs{1}(:) - aprxImg(:))^2;
            %
            [coefs2,scales2]   = step(analyzer,srcImgs{2},nLevels);            
            coefs2 = coefs2.*mask2;
            aprxImg = step(synthesizer,coefs2,scales2);
            costExpctd = costExpctd + norm(srcImgs{2}(:) - aprxImg(:))^2;            
            
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
        
        % Test for default construction
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
            import saivdr.dictionary.nsolt.design.*
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels,...
                'IsFixedCoefs',false);

            % Pre
            import saivdr.dictionary.nsolt.*
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
 
        % Test for default construction
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
            import saivdr.dictionary.nsolt.design.*
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels,...
                'IsFixedCoefs',true);

            % Pre
            import saivdr.dictionary.nsolt.*
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

        % Test for default construction
        function testAprxErrDec22Ch5plus2Ord44Lv2MultiImgsFixedCoefs(testCase)
            
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
            import saivdr.dictionary.nsolt.*
            import saivdr.dictionary.nsolt.design.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs, ...
                'NumberOfChannels', nChs, ...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments',nVm,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            synthesizer = NsoltFactory.createSynthesisSystem(lppufb,...
                'BoundaryOperation','Termination');
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels,...
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
            
            % Actual values
            costActual = step(testCase.aprxerr,lppufb,sparseCoefs,setOfScales);
            
            % 
            diff = norm(costExpctd-costActual)/norm(costExpctd);
            testCase.verifyEqual(costExpctd,costActual,'RelTol',1e-10,sprintf('%g',diff));
            
        end

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
            import saivdr.dictionary.nsolt.design.*
            testCase.aprxerr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels,...
                'IsFixedCoefs',false);

            % Pre
            import saivdr.dictionary.nsolt.*
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
 
    end
end
