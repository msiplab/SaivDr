classdef OvsdLpPuFb2dTypeICostEvaluatorTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB2dTYPEICOSTEVALUATORTESTCASE Test case for OvsdLpPuFb2dTypeICostEvaluator
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
            lppufbExpctd = OvsdLpPuFb2dTypeIVm1System(...  
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
            lppufbExpctd = OvsdLpPuFb2dTypeIVm1System(...
                'NumberOfChannels',nChs,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
                'LpPuFb',lppufbExpctd);
            
            % Actual value
            lppufbActual = get(testCase.evaluator,'LpPuFb');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end

        % Test for default construction
        function testInverseBlockDct(testCase)
            
            nDecs   = [2 2];
            height  = 12;
            width   = 16;
            nLevels = 1;
            srcImg  = rand(height,width);
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs);
            angs = get(lppufb,'Angles');
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord00(testCase)
            
            nDecs  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
                        
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord00Vm0(testCase)
            
            nDecs  = [2 2];
            nOrds  = [0 0];  
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms = 0;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord00Vm0PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nOrds  = [0 0];  
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms = 0;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
               'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord00Vm1(testCase)
            
            nDecs  = [2 2];
            nOrds  = [0 0];  
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord00Vm1PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nOrds  = [0 0];  
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
       function testInverseBlockDctDec44(testCase)
            
            nDecs  = [4 4];
            height = 24;
            width  = 32;
            srcImg  = rand(height,width);
            nLevels = 1;
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs);
            angs  = get(lppufb,'Angles');
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch44Ord00Vm1(testCase)
            
            nDecs  = [2 2];
            nChs   = [4 4];
            nOrds  = [0 0];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms    = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch44Ord00Vm1PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nChs   = [4 4];
            nOrds  = [0 0];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms    = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch44Ord00Vm0(testCase)
            
            nDecs  = [2 2];
            nChs   = [4 4];
            nOrds  = [0 0];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms    = 0;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch44Ord00Vm0PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nChs   = [4 4];
            nOrds  = [0 0];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms    = 0;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord22Vm0PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nOrds  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms   = 0;
            isPext = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord22Vm1PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nOrds  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms   = 1;
            isPext = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord22Vm0(testCase)
            
            nDecs  = [2 2];
            nOrds  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms   = 0;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord22Vm1(testCase)
            
            nDecs  = [2 2];
            nOrds  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms   = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord22(testCase)
            
            nDecs  = [2 2];
            nOrds  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch44Ord22(testCase)
            
            nDecs  = [2 2];
            nChs   = [4 4];
            nOrds  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22h44Ord22Vm0PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nChs   = [4 4];
            nOrds  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms   = 0;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22h44Ord22Vm0(testCase)
            
            nDecs  = [2 2];
            nChs   = [4 4];
            nOrds  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms   = 0;
            isPext  = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22h44Ord22Vm0Clone(testCase)
            
            nDecs  = [2 2];
            nChs   = [4 4];
            nOrds  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms   = 0;
            isPext  = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            evaluator_ = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch44Ord22Vm1PeriodicExt(testCase)  
            
            nDecs  = [2 2];
            nChs   = [4 4];
            nOrds  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms  = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch44Ord22Vm1PeriodicExtClone(testCase)  
            
            nDecs  = [2 2];
            nChs   = [4 4];
            nOrds  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms  = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            evaluator_ = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch44Ord22Vm1(testCase)  
            
            nDecs  = [2 2];
            nChs   = [4 4];
            nOrds  = [2 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms  = 1;
            isPext  = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord44(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [4 4];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord44PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [4 4];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord44(testCase)
            
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [4 4];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord44Clone(testCase)
            
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [4 4];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            evaluator_ = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord44PeriodicExt(testCase)
            
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [4 4];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord44PeriodicExtClone(testCase)
            
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [4 4];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            evaluator_ = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord66(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [6 6];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord66PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [6 6];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord66(testCase)
            
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [6 6];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord66PeriodicExt(testCase)
            
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [6 6];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord02(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [0 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord02(testCase)
            
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [0 2];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord04(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [0 4];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord04(testCase)
        
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [0 4];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord20(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [2 0];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord20(testCase)
            
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [2 0];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord40(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [4 0];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord40(testCase)
                      
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [4 0];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord02PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [0 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord02PeriodicExt(testCase)
            
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [0 2];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord04PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [0 4];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord04PeriodicExt(testCase)
        
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [0 4];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord20PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [2 0];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord20PeriodicExt(testCase)
            
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [2 0];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord40PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [4 0];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular',...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord40PeriodicExt(testCase)
                      
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [4 0];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord24(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [2 4];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord24(testCase)
            
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [2 4];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord24PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [2 4];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord24PeriodicExt(testCase)
        
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [2 4];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord42(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [4 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord42(testCase)
            
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [4 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec22Ch22Ord42PeriodicExt(testCase)
            
            nDecs  = [2 2];
            nChs   = [2 2];
            nOrds  = [4 2];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec44Ch88Ord42PeriodicExt(testCase)
        
            nDecs  = [4 4];
            nChs   = [8 8];
            nOrds  = [4 2];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            isPext  = true;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'NumberOfChannels',nChs,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        
        function testSetLpPuFb2dDec22Ch44Ord44(testCase)
            
            nDecs = [2 2];
            nChs  = [4 4];
            nOrds = [4 4];
            height = 24;
            width  = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            
            % Instantiation of target class
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular', ...
                'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,...
                'BoundaryOperation','Circular');
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstPre = sum((srcImg(:)-recImg(:)).^2);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
            ch = [ 4 4 ];
            ord = 4;
            height = 24;
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
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.evaluator = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
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
            testCase.evaluator = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
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
            
            nDecs   = [ 2 2 ];
            nChs    = [ 4 4 ];
            nOrds   = [ 4 4 ];
            height  = 64;
            width   = 64;
            srcImg  = rand(height,width);
            nLevels = 1;
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec12Ch22Ord22Vm0(testCase)
            
            nDecs = [ 1 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms   = 0;
            isPext = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec21Ch22Ord22Vm0(testCase)
            
            nDecs = [ 1 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms    = 0;
            isPext  = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb, 'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec12Ch22Ord22Vm1(testCase)
            
            nDecs = [ 1 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms   = 1;
            isPext = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
        function testStepDec21Ch22Ord22Vm1(testCase)
            
            nDecs = [ 1 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            height = 12;
            width  = 16;
            srcImg = rand(height,width);
            nLevels = 1;
            nVms    = 1;
            isPext  = false;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds,...
                'NumberOfVanishingMoments',nVms);
            angs = get(lppufb,'Angles');
            angs = pi/6*randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'NumberOfLevels',nLevels);
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb);
            [coefs,scales] = step(analyzer,srcImg);
            [~,idxs] = sort(abs(coefs));
            coefs(idxs(1:floor(height*width/2)))=0;
            recImg = step(synthesizer,coefs,scales);
            cstExpctd = sum((srcImg(:)-recImg(:)).^2);
            grdExpctd = testCase.gradient(lppufb,srcImg,coefs,scales,...
                isPext);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.evaluator = OvsdLpPuFb2dTypeICostEvaluator(...
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
                delta  = 1e-6;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            synCur = NsoltFactory.createSynthesis2dSystem(lppufb,...
                'IsCloneLpPuFb',true);
            if isPext
                set(synCur,'BoundaryOperation','Circular');
            end            
            recImgCur = step(synCur,coefs,scales);
            evlCur    = sum((srcImg(:)-recImgCur(:)).^2);
            angs      = get(lppufb,'Angles');
            
            % Numerical gradient
            clonefb = clone(lppufb);
            synDlt = NsoltFactory.createSynthesis2dSystem(clonefb,...
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

