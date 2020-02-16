classdef OvsdLpPuFb3dTypeICostEvaluatorTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB3DTYPEICOSTEVALUATORTESTCASE Test case for OvsdLpPuFb3dTypeICostEvaluator
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
        evaluator
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.evaluator);
        end
    end
    
    methods (Test)
        
        % Test
        function testDefaultConstruction(testCase)
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufbExpctd = OvsdLpPuFb3dTypeIVm1System(...  
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufbExpctd);
            
            % Actual value
            lppufbActual = get(testCase.evaluator,'LpPuFb');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);

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
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufbExpctd);
            
            % Actual value
            lppufbActual = get(testCase.evaluator,'LpPuFb');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end

        % Test for default construction
        function testInverseBlockDct(testCase)
            
            nDecs   = [2 2 2];
            height  = 12;
            width   = 16;
            depth   = 20;
            nLevels = 1;
            srcImg  = rand(height,width,depth);
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs);
            angs = get(lppufb,'Angles');
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);            
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));            
        end
        
        % Test
        function testStepDec222Ch44Ord000(testCase)
            
            nDecs  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
                        
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);            
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff)); 
            
        end

        %Test
        function testStepDec222Ch44Ord000Vm0(testCase)
            
            nDecs  = [2 2 2];
            nOrds  = [0 0 0];  
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms = 0;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
            
        end

        %Test
        function testStepDec222Ch44Ord000Vm0PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nOrds  = [0 0 0];  
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms = 0;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
            
        end

        %Test
        function testStepDec222Ch44Ord000Vm1(testCase)
            
            nDecs  = [2 2 2];
            nOrds  = [0 0 0];  
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
            
        end

        %Test
        function testStepDec222Ch44Ord000Vm1PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nOrds  = [0 0 0];  
            height = 12;
            width  = 16;
            depth  = 20; 
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
            
        end

       % Test
           function testInverseBlockDctDec444(testCase)
            
            nDecs  = [4 4 4];
            height = 24;
            width  = 32;
            depth  = 40;
            srcImg  = rand(height,width,depth);
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs);
            angs  = get(lppufb,'Angles');
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
            
           end
           
           % Test
           function testStepDec222Ch55Ord000Vm1(testCase)
               
               nDecs  = [2 2 2];
               nChs   = [5 5];
               nOrds  = [0 0 0];
               height = 24;
               width  = 32;
               depth  = 40;
               srcImg = rand(height,width,depth);
               nLevels = 1;
               nVms    = 1;
               isPext  = true;
               
               % Preparation
               import saivdr.dictionary.nsoltx.*
               lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                   'DecimationFactor',nDecs,...
                   'NumberOfChannels',nChs,...
                   'PolyPhaseOrder',nOrds,...
                   'NumberOfVanishingMoments',nVms);
               angs = get(lppufb,'Angles');
               angs = pi/6*randn(size(angs));
               set(lppufb,'Angles',angs);
               
               % Expected values
               analyzer    = NsoltFactory.createAnalysis3dSystem(...
                   lppufb, 'NumberOfLevels', nLevels);
               synthesizer = NsoltFactory.createSynthesis3dSystem(...
                   lppufb);
               [coefs,scales] = step(analyzer,srcImg);
               [~,idxs] = sort(abs(coefs));
               coefs(idxs(1:floor(height*width*depth/2)))=0;
               recImg = step(synthesizer,coefs,scales);
               cstExpctd = sum((srcImg(:)-recImg(:)).^2);
               grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                   isPext);
               
               % Instantiation of target class
               import saivdr.dictionary.nsoltx.design.*
               testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                   'LpPuFb',lppufb);
               
               % Actual values
               [cstActual,grdActual] = ...
                   step(testCase.evaluator,srcImg,coefs,scales);
               
               % Evaluation
               diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
               testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                   sprintf('%g',diff));
               testCase.verifySize(grdActual,[numel(angs) 1]);
               diff = max(abs(grdExpctd(:) - grdActual(:)));
               testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                   sprintf('%g',diff));
           end
           
        % Test
        function testStepDec222Ch55Ord000Vm1PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [5 5];
            nOrds  = [0 0 0];
            height = 24;
            width  = 32;
            depth  = 40;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms    = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');            
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
        end
  
        % Test
        function testStepDec222Ch544Ord000Vm0(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [5 5];
            nOrds  = [0 0 0];
            height = 24;
            width  = 32;
            depth  = 40;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms    = 0;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
        end

        % Test
        function testStepDec222Ch55Ord000Vm0PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [5 5];
            nOrds  = [0 0 0];
            height = 24;
            width  = 32;
            depth  = 40;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms    = 0;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');            
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
        end

        % Test
        function testStepDec222Ch44Ord222Vm0PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms   = 0;
            isPext = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');            
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord222Vm1PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms   = 1;
            isPext = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');            
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
        end

        % Test
        function testStepDec222Ch44Ord222Vm0(testCase)
            
            nDecs  = [2 2 2];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms   = 0;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);            
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch44Ord222Vm1(testCase)
            
            nDecs  = [2 2 2];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms   = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);            
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
        end

        % Test
        function testStepDec222Ch44Ord222(testCase)
            
            nDecs  = [2 2 2];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);            
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));            
        end

        % Test
        function testStepDec222Ch55Ord222(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [5 5];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);            
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));  
        end

        % Test
        function testStepDec222h55Ord222Vm0PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [5 5];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms   = 0;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222h55Ord222Vm0(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [5 5];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms   = 0;
            isPext  = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
                
        % Test
        function testStepDec222h55Ord222Vm0Clone(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [5 5];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms   = 0;
            isPext  = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            evaluator_ = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            testCase.evaluator = clone(evaluator_);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

        % Test
        function testStepDec222Ch55Ord222Vm1PeriodicExt(testCase)  
            
            nDecs  = [2 2 2];
            nChs   = [5 5];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms  = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
          
        % Test
        function testStepDec222Ch55Ord222Vm1PeriodicExtClone(testCase)  
            
            nDecs  = [2 2 2];
            nChs   = [5 5];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms  = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            evaluator_ = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            testCase.evaluator = clone(evaluator_);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
         
        % Test
        function testStepDec222Ch55Ord222Vm1(testCase)  
            
            nDecs  = [2 2 2];
            nChs   = [5 5];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms  = 1;
            isPext  = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
                
        % Test
        function testStepDec222Ch44Ord444(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [4 4 4];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch44Ord444PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [4 4 4];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch66Ord222(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch66Ord222Clone(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            evaluator_ = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            testCase.evaluator = clone(evaluator_);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));
        end
        
        % Test
        function testStepDec222Ch66Ord222PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [2 2 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch66Ord444PeriodicExtClone(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [4 4 4];
            height = 12;
            width  = 16;
            depth = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            evaluator_ = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            testCase.evaluator = clone(evaluator_);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch44Ord666(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [6 6 6];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch44Ord666PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [6 6 6];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

        % Test
        function testStepDec222Ch66Ord444(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [4 4 4];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch66Ord666PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [6 6 6];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

        % Test
        function testStepDec222Ch44Ord002(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [0 0 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch66Ord002(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [0 0 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

        % Test
        function testStepDec222Ch44Ord004(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [0 0 4];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

        % Test
        function testStepDec222Ch66Ord004(testCase)
        
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [0 0 4];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch44Ord020(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [0 2 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch66Ord020(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [0 2 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch44Ord040(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [0 4 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch66Ord040(testCase)
                      
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [0 4 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch44Ord002PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [0 0 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

        % Test
        function testStepDec222Ch66Ord002PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [0 0 2];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

        % Test
        function testStepDec222Ch44Ord004PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [0 0 4];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

        % Test
        function testStepDec222Ch66Ord004PeriodicExt(testCase)
        
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [0 0 4];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch44Ord020PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [0 2 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch66Ord020PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [0 2 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch44Ord040PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [0 4 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch66Ord040PeriodicExt(testCase)
                      
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [0 4 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

        % Test
        function testStepDec222Ch44Ord200(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [2 0 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch66Ord200(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [2 0 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch44Ord200PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [2 0 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

        % Test
        function testStepDec222Ch66Ord200PeriodicExt(testCase)
        
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [2 0 0 ];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end   

        % Test
        function testStepDec222Ch44Ord420(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4 ];
            nOrds  = [4 2 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch66Ord420(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [4 2 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec222Ch44Ord420PeriodicExt(testCase)
            
            nDecs  = [2 2 2];
            nChs   = [4 4];
            nOrds  = [4 2 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
                                    
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

        % Test
        function testStepDec222Ch66Ord420PeriodicExt(testCase)
        
            nDecs  = [2 2 2];
            nChs   = [6 6];
            nOrds  = [4 2 0];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        function testSetLpPuFb3dDec222Ch88Ord444(testCase)
            
            nDecs = [2 2 2];
            nChs  = [8 8];
            nOrds = [4 4 4];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            
            % Instantiation of target class
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstPre = sum((srcImg(:)-recImg(:)).^2);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Termination');
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            cstPst = step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = sum((cstPst(:)-cstPre(:)).^2);
            testCase.verifyTrue(diff<1e-15);
            
            % ReInstantiation of target class
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb,...
                'BoundaryOperation','Termination');
            cstPst = step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = sum((cstPst(:)-cstPre(:)).^2);
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
        function testIsCloneLpPuFbFalse(testCase)
            
            dec = 2;
            ch = [ 6 6 ];
            ord = 4;
            height = 24;
            width = 32;
            depth = 40;
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
            subCoefs{11} = rand(height/(dec),width/(dec),depth/(dec));        
            subCoefs{12} = rand(height/(dec),width/(dec),depth/(dec));                
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
                'PolyPhaseOrder',[ord ord ord],...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.evaluator = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'IsCloneLpPuFb',true);
            
            % Pre
            imgPre = step(testCase.evaluator,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Pst
            imgPst = step(testCase.evaluator,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = sum((imgPst(:)-imgPre(:)).^2);
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,sprintf('%g',diff));
            
            % ReInstantiation of target class
            testCase.evaluator = NsoltSynthesis3dSystem(...
                'LpPuFb3d',lppufb,...
                'IsCloneLpPuFb',false);
            
            % Pre
            imgPre = step(testCase.evaluator,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Pst
            imgPst = step(testCase.evaluator,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = sum((imgPst(:)-imgPre(:)).^2);
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end
        
        % Test
        function testClone(testCase)
            
            nDecs   = [ 2 2 2 ];
            nChs    = [ 4 4 ];
            nOrds   = [ 4 4 4 ];
            height  = 64;
            width   = 64;
            depth   = 64;
            srcImg  = rand(height,width,depth);
            nLevels = 1;
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Clone
            cloneevaluator = clone(testCase.evaluator);
            
            % Evaluation
            testCase.verifyEqual(cloneevaluator,testCase.evaluator);
            testCase.verifyFalse(cloneevaluator == testCase.evaluator);
            prpOrg = get(testCase.evaluator,'LpPuFb');
            prpCln = get(cloneevaluator,'LpPuFb');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            [cstExpctd,grdExpctd] = step(testCase.evaluator,srcImg,coefs,scales);
            [cstActual,grdActual] = step(cloneevaluator,srcImg,coefs,scales);
            testCase.assertEqual(cstActual,cstExpctd);
            testCase.assertEqual(grdActual,grdExpctd);
            
        end
        
        %Test
        function testStepDec112Ch22Ord222Vm0(testCase)
            
            nDecs = [ 1 1 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 2 ];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms   = 0;
            isPext = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec221Ch22Ord222Vm0(testCase)
            
            nDecs = [ 2 2 1 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 2 ];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms    = 0;
            isPext  = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

        %Test
        function testStepDec211Ch22Ord222Vm1(testCase)
            
            nDecs = [ 2 1 1 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 2 ];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms   = 1;
            isPext = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end
        
        % Test
        function testStepDec121Ch22Ord222Vm1(testCase)
            
            nDecs = [ 1 2 1 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 2 ];
            height = 12;
            width  = 16;
            depth  = 20;
            srcImg = rand(height,width,depth);
            nLevels = 1;
            nVms    = 1;
            isPext  = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb, 'NumberOfLevels', nLevels);
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width*depth/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb3dTypeICostEvaluator(...
                'LpPuFb',lppufb);
            
            % Actual values
            [cstActual,grdActual] = ...
                step(testCase.evaluator,srcImg,coefs,scales);
            
            % Evaluation
            diff = max(abs(cstExpctd - cstActual)./(abs(cstExpctd)));
            testCase.verifyEqual(cstActual,cstExpctd,'RelTol',1e-10,...
                sprintf('%g',diff));
            testCase.verifySize(grdActual,[numel(angs) 1]);
            diff = max(abs(grdExpctd(:) - grdActual(:)));
            testCase.verifyEqual(grdActual,grdExpctd,'AbsTol',1e-3,...
                sprintf('%g',diff));              
        end

    end

    methods (Access = public, Static = true)
        
        function grad = gradient(lppufb,srcImg,coefs,scales,isPext,delta)

            if nargin < 5
                isPext = false;
            end
            if nargin < 6
                delta  = 1e-8;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            synCur = NsoltFactory.createSynthesis3dSystem(lppufb,...
                'IsCloneLpPuFb',true);
            if isPext
                set(synCur,'BoundaryOperation','Circular');
            end            
            recImgCur = step(synCur,coefs,scales);
            evlCur    = sum((srcImg(:)-recImgCur(:)).^2);
            angs      = get(lppufb,'Angles');
            
            % Numerical gradient
            clonefb = clone(lppufb);
            synDlt = NsoltFactory.createSynthesis3dSystem(clonefb,...
                'IsCloneLpPuFb',false);
            if isPext
                set(synDlt,'BoundaryOperation','Circular');
            end
            nAngs = numel(angs);
            grad  = zeros(nAngs,1);
            for iAng = 1:nAngs
                angs_delta = angs;
                angs_delta(iAng) = angs(iAng) + delta;
                set(clonefb,'Angles',angs_delta);
                recImgDlt = step(synDlt,coefs,scales);
                evlDlt = sum((srcImg(:)-recImgDlt(:)).^2);
                grad(iAng) = (evlDlt-evlCur)./delta;
            end
        end

    end
end

