classdef NsoltDictionaryLearningTestCase < matlab.unittest.TestCase
    %NSOLTDICTIONARYLEARNINGTESTCASE Test case for NsoltDictionaryLearning
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
    
    properties (TestParameter)
        %useparallel = struct('true', true, 'false', false );
        %usegpu = struct('true', true, 'false', false );        
        %niter = struct('small',1, 'large', 4 );
        %dsplit = struct('small',1, 'large', 2);
        %vsplit = struct('small',1, 'large', 2);        
        %hsplit = struct('small',1, 'large', 2);                
        %nlevels = { 1, 3 };
    end
    
    properties
        designer
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.designer);
        end
    end
    
    methods (Test)
        
        
        
        % Test 
        function testNsoltDictionaryLearningGpDec22Ch62Ord44(testCase)
    
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = [ 6 2 ];
            nOrds = [ 4 4 ];
            nSprsCoefs = 4;
            isOptMus = false;
            srcImgs{1} = rand(16,16);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'TrainingImages',srcImgs,...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'NumbersOfPolyphaseOrder',nOrds);
            
            % Pre
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufbPre);
            analyzer  = NsoltFactory.createAnalysis2dSystem(lppufbPre,...
                'NumberOfLevels',nLevels);
            import saivdr.sparserep.*
            gpnsolt = GradientPursuit(...
                'Dictionary', { synthesizer, analyzer });
            gpnsolt.NumberOfSparseCoefficients = nSprsCoefs;
            [~, coefsPre{1},scales{1}] = gpnsolt.step(srcImgs{1});
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufbPre,coefsPre,scales);
            
            % Pst
            options = optimoptions('fminunc');
            options = optimoptions(options,'Algorithm','quasi-newton');
            options = optimoptions(options,'Display','off');
            options = optimoptions(options,'MaxIter',2);
            [~, costPst] = step(testCase.designer,options,isOptMus);

            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            testCase.verifyThat(costPst, IsLessThan(costPre));            

        end
        
        % Test 
        function testNsoltDictionaryLearningGpDec22Ch62Ord44Ga(testCase)
    
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = [ 6 2 ];
            nOrds = [ 4 4 ];
            nSprsCoefs = 4;
            isOptMus = true;
            srcImgs{1} = rand(16,16);            
            optfcn = @ga;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'TrainingImages',srcImgs,...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2);
            
            % Pre
            import saivdr.sparserep.*
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysis2dSystem(lppufbPre,...
                'NumberOfLevels',nLevels);            
            gpnsolt = GradientPursuit(...
                'Dictionary', { synthesizer, analyzer });
            gpnsolt.NumberOfSparseCoefficients = nSprsCoefs;
            [~, coefsPre{1}, scales{1}] = gpnsolt.step(srcImgs{1});
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufbPre,coefsPre,scales);
            
            % Pst
            angles = get(lppufbPre,'Angles');            
            options = gaoptimset(optfcn);
            options = gaoptimset(options,'Display','off');
            options = gaoptimset(options,'PopulationSize',10);            
            options = gaoptimset(options,'Generations',1);            
            options = gaoptimset(options,'PopInitRange',...
                [angles(:).'-pi;angles(:).'+pi]);
            options = gaoptimset(options,'UseParallel','always');
            %
            [~, costPst] = step(testCase.designer,options,isOptMus);


            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(costPst, IsLessThan(costPre));            

        end
        
        % Test
        function testNsoltDictionaryLearningIhtDec22Ch62Ord44(testCase)
            
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = [ 6 2 ];
            nOrds = [ 4 4 ];
            nSprsCoefs = 4;
            isOptMus = false;
            srcImgs{1} = rand(16,16);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'TrainingImages',srcImgs,...
                'SparseApproximation','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'NumbersOfPolyphaseOrder',nOrds);
            
            % Pre
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufbPre);
            analyzer  = NsoltFactory.createAnalysis2dSystem(lppufbPre,...
                'NumberOfLevels',nLevels);
            import saivdr.sparserep.*
            gpnsolt = IterativeHardThresholding(...
                'Dictionary', { synthesizer, analyzer });
            gpnsolt.NumberOfSparseCoefficients = nSprsCoefs;
            [~, coefsPre{1},scales{1}] = gpnsolt.step(srcImgs{1});
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufbPre,coefsPre,scales);
            
            % Pst
            options = optimoptions('fminunc');
            options = optimoptions(options,'Algorithm','quasi-newton');
            options = optimoptions(options,'Display','off');
            options = optimoptions(options,'MaxIter',2);
            [~, costPst] = step(testCase.designer,options,isOptMus);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
                
        % Test
        function testNsoltDictionaryLearningIhtDec22Ch44Ord44Sgd(testCase)
            
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ]; 
            nSprsCoefs = 4;
            isOptMus = false;
            srcImgs{1} = imfilter(rand(12,16),ones(2)/4);
            srcImgs{2} = imfilter(rand(12,16),ones(2)/4);
            srcImgs{3} = imfilter(rand(12,16),ones(2)/4);
            srcImgs{4} = imfilter(rand(12,16),ones(2)/4);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'TrainingImages',srcImgs,...
                'SparseApproximation','IterativeHardThresholding',...
                'DictionaryUpdater','NsoltDictionaryUpdateSgd',...
                'IsFixedCoefs',true,...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'NumbersOfPolyphaseOrder',nOrds,...
                'GradObj', 'on');
            
            % Pre
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufbPre);
            analyzer  = NsoltFactory.createAnalysis2dSystem(lppufbPre,...
                'NumberOfLevels',nLevels);
            import saivdr.sparserep.*
            ihtnsolt = IterativeHardThresholding(...
                'Dictionary', { synthesizer, analyzer });
            nImgs = length(srcImgs);
            coefsPre = cell(nImgs,1);
            setOfScales   = cell(nImgs,1);
            ihtnsolt.NumberOfSparseCoefficients = nSprsCoefs;
            for iImg = 1:nImgs
                [~, coefsPre{iImg},setOfScales{iImg}] = ...
                    ihtnsolt.step(srcImgs{iImg});
            end
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufbPre,coefsPre,setOfScales);
            
            % Pst
            options = optimset(...
                'MaxIter',2*nImgs,...
                'TolX',1e-4);                
%             for iter = 1:5
%                 step(testCase.designer,options,isOptMus);
%             end
            [~, costPst] = step(testCase.designer,options,isOptMus);            
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        % Test
        function testNsoltDictionaryLearningIhtDec22Ch44Ord44GradObj(testCase)
            
            % Parameter settings
            nLevels = 1;
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nSprsCoefs = 4;
            isOptMus = false;
            srcImgs{1} = rand(12,16);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'TrainingImages',srcImgs,...
                'SparseApproximation','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nSprsCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'NumbersOfPolyphaseOrder',nOrds,...
                'IsFixedCoefs',true,...
                'GradObj', 'on');
            
            % Pre
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufbPre);
            analyzer  = NsoltFactory.createAnalysis2dSystem(lppufbPre,...
                'NumberOfLevels',nLevels);
            import saivdr.sparserep.*
            gpnsolt = IterativeHardThresholding(...
                'Dictionary', { synthesizer, analyzer });
            gpnsolt.NumberOfSparseCoefficients = nSprsCoefs;
            [~, coefsPre{1},scales{1}] = gpnsolt.step(srcImgs{1});
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufbPre,coefsPre,scales);
            
            % Pst
            options = optimoptions('fminunc');
            options = optimoptions(options,'Algorithm','trust-region');
            options = optimoptions(options,'Display','off');
            options = optimoptions(options,'MaxIter',2);
            options = optimoptions(options,'GradObj','on');
            [~, costPst] = step(testCase.designer,options,isOptMus);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        % Test 
        function testNsoltDictionaryLearningIhtDec22Ch62Ord44Ga(testCase)
    
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = [ 6 2 ];
            nOrds = [ 4 4 ];
            nSprsCoefs = 4;
            isOptMus = true;
            srcImgs{1} = rand(16,16);            
            optfcn = @ga;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'TrainingImages',srcImgs,...
                'SparseApproximation','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2);
            
            % Pre
            import saivdr.sparserep.*
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysis2dSystem(lppufbPre,...
                'NumberOfLevels',nLevels);            
            gpnsolt = IterativeHardThresholding(...
                'Dictionary', { synthesizer, analyzer });
            gpnsolt.NumberOfSparseCoefficients = nSprsCoefs;
            [~, coefsPre{1}, scales{1}] = gpnsolt.step(srcImgs{1});
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufbPre,coefsPre,scales);
            
            % Pst
            angles = get(lppufbPre,'Angles');
            options = gaoptimset(optfcn);
            options = gaoptimset(options,'Display','off');
            options = gaoptimset(options,'PopulationSize',10);
            options = gaoptimset(options,'Generations',1);
            options = gaoptimset(options,'PopInitRange',...
                [angles(:).'-pi;angles(:).'+pi]);
            options = gaoptimset(options,'UseParallel','always');
            %
            [~, costPst] = step(testCase.designer,options,isOptMus);
            
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
        % Test
        function testNsoltDictionaryLearningGpDec222Ch55Ord222(testCase)
            
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = [ 5 5 ];
            nOrds = [ 2 2 2 ];
            nSprsCoefs = 4;
            isOptMus = false;
            srcImgs{1} = rand(16,16,16);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'NumberOfDimensions','Three',...
                'TrainingImages',srcImgs,...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'NumbersOfPolyphaseOrder',nOrds);
            
            % Pre
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer  = NsoltFactory.createAnalysis3dSystem(lppufbPre);
            analyzer.NumberOfLevels = nLevels;
            import saivdr.sparserep.*
            gpnsolt = GradientPursuit(...
                'Dictionary', { synthesizer, analyzer });
            gpnsolt.NumberOfSparseCoefficients = nSprsCoefs;
            [~, coefsPre{1},scales{1}] = gpnsolt.step(srcImgs{1});
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufbPre,coefsPre,scales);
            
            % Pst
            options = optimoptions('fminunc');
            options = optimoptions(options,'Algorithm','quasi-newton');
            options = optimoptions(options,'Display','off');
            options = optimoptions(options,'MaxIter',2);
            [~, costPst] = step(testCase.designer,options,isOptMus);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end
        
                
        % Test 
        function testNsoltDictionaryLearningGpDec222Ch64Ord222Ga(testCase)
    
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = [ 6 4 ];
            nOrds = [ 2 2 2 ];
            nSprsCoefs = 4;
            isOptMus = true;
            srcImgs{1} = rand(16,16,16);            
            optfcn = @ga;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'NumberOfDimensions','Three',...
                'TrainingImages',srcImgs,...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2);
            
            % Pre
            import saivdr.sparserep.*
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysis3dSystem(lppufbPre);
            analyzer.NumberOfLevels = nLevels;            
            gpnsolt = GradientPursuit(...
                'Dictionary', { synthesizer, analyzer });
            gpnsolt.NumberOfSparseCoefficients = nSprsCoefs;
            [~, coefsPre{1}, scales{1}] = gpnsolt.step(srcImgs{1});
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufbPre,coefsPre,scales);
            
            % Pst
            angles = get(lppufbPre,'Angles');            
            options = gaoptimset(optfcn);
            options = gaoptimset(options,'Display','off');
            options = gaoptimset(options,'PopulationSize',10);            
            options = gaoptimset(options,'Generations',1);            
            options = gaoptimset(options,'PopInitRange',...
                [angles(:).'-pi;angles(:).'+pi]);
            options = gaoptimset(options,'UseParallel','always');
            %
            [~, costPst] = step(testCase.designer,options,isOptMus);

            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(costPst, IsLessThan(costPre));            

        end

     % Test 
        function testNsoltDictionaryLearningIhtDec222Ch64Ord222(testCase)
    
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = [ 6 4 ];
            nOrds = [ 2 2 2 ];
            nSprsCoefs = 4;
            isOptMus = false;
            srcImgs{1} = rand(16,16,16);            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'NumberOfDimensions','Three',...
                'TrainingImages',srcImgs,...
                'SparseApproximation','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'NumbersOfPolyphaseOrder',nOrds);
            
            % Pre
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer  = NsoltFactory.createAnalysis3dSystem(lppufbPre,...
                'NumberOfLevels',nLevels);
            import saivdr.sparserep.*
            gpnsolt = IterativeHardThresholding(...
                'Dictionary', { synthesizer, analyzer });
            gpnsolt.NumberOfSparseCoefficients = nSprsCoefs;
            [~, coefsPre{1},scales{1}] = gpnsolt.step(srcImgs{1});
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufbPre,coefsPre,scales);
            
            % Pst
            options = optimoptions('fminunc');
            options = optimoptions(options,'Algorithm','quasi-newton');
            options = optimoptions(options,'Display','off');
            options = optimoptions(options,'MaxIter',2);
            [~, costPst] = step(testCase.designer,options,isOptMus);

            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(costPst, IsLessThan(costPre));            

        end        
        
      % Test 
        function testNsoltDictionaryLearningIhtDec222Ch55Ord222Ga(testCase)
    
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = [ 5 5 ];
            nOrds = [ 2 2 2 ];
            nSprsCoefs = 4;
            isOptMus = true;
            srcImgs{1} = rand(16,16,16);            
            optfcn = @ga;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'NumberOfDimensions','Three',...
                'TrainingImages',srcImgs,...
                'SparseApproximation','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2);
            
            % Pre
            import saivdr.sparserep.*
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysis3dSystem(lppufbPre,...
                'NumberOfLevels',nLevels);            
            gpnsolt = IterativeHardThresholding(...
                'Dictionary', { synthesizer, analyzer });
            gpnsolt.NumberOfSparseCoefficients = nSprsCoefs;
            [~, coefsPre{1}, scales{1}] = gpnsolt.step(srcImgs{1});
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs,...
                'NumberOfLevels',nLevels);
            costPre = step(aprxErr,lppufbPre,coefsPre,scales);
            
            % Pst
            angles = get(lppufbPre,'Angles');
            options = gaoptimset(optfcn);
            options = gaoptimset(options,'Display','off');
            options = gaoptimset(options,'PopulationSize',10);
            options = gaoptimset(options,'Generations',1);
            options = gaoptimset(options,'PopInitRange',...
                [angles(:).'-pi;angles(:).'+pi]);
            options = gaoptimset(options,'UseParallel','always');
            %
            [~, costPst] = step(testCase.designer,options,isOptMus);
            
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifyThat(costPst, IsLessThan(costPre));
            
        end        
        
        % Test 
        function testNsoltDictionaryLearningIhtDec22Ch44Ord22(testCase)
    
            % Parameter settings
            nCoefs   = 4;
            nLevels  = 1;
            nChs     = [ 4 4 ];
            nOrds    = [ 2 2 ];
            srcImgs{1} = rand(16,16);
            optfcn = @fminunc;
            nUnfixedSteps = 1;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'IsFixedCoefs',true,...
                'NumberOfUnfixedInitialSteps', nUnfixedSteps,...
                'NumberOfDimensions','Two',...
                'TrainingImages',srcImgs,...
                'SparseApproximation','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds);

            % Options
            options = optimoptions(optfcn);
            options = optimoptions(options,'Algorithm','quasi-newton');
            options = optimoptions(options,'Display','off');
            options = optimoptions(options,'MaxIter',2);
            
            % State after Step 1
            step(testCase.designer,options,[]);
            stateActual = get(testCase.designer,'IsPreviousStepFixed');
            countActual = get(testCase.designer,'Count');            
            testCase.verifyFalse(stateActual);
            testCase.verifyEqual(countActual,2);
                        
            % State after Step 2
            step(testCase.designer,options,[]);
            stateActual = get(testCase.designer,'IsPreviousStepFixed');            
            countActual = get(testCase.designer,'Count');            
            testCase.verifyTrue(stateActual);
            testCase.verifyEqual(countActual,3);            

        end        
        
        
        % Test 
        function testNsoltDictionaryLearningIhtDec222Ch55Ord222(testCase)
    
            % Parameter settings
            nCoefs   = 4;
            nLevels  = 1;
            nChs     = [ 5 5];
            nOrds    = [ 2 2 2 ];
            srcImgs{1} = rand(16,16,16);
            optfcn = @fminunc;
            nUnfixedSteps = 2;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'IsFixedCoefs',true,...
                'NumberOfUnfixedInitialSteps',nUnfixedSteps,...
                'NumberOfDimensions','Three',...
                'TrainingImages',srcImgs,...
                'SparseApproximation','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds);

            % Options
            options = optimoptions(optfcn);
            options = optimoptions(options,'Algorithm','quasi-newton');
            options = optimoptions(options,'Display','off');
            options = optimoptions(options,'MaxIter',2);
            
            % State after Step 1
            step(testCase.designer,options,[]);
            stateActual = get(testCase.designer,'IsPreviousStepFixed');
            countActual = get(testCase.designer,'Count');            
            testCase.verifyFalse(stateActual);
            testCase.verifyEqual(countActual,2);
                        
            % State after Step 2
            step(testCase.designer,options,[]);
            stateActual = get(testCase.designer,'IsPreviousStepFixed');            
            countActual = get(testCase.designer,'Count');            
            testCase.verifyFalse(stateActual);
            testCase.verifyEqual(countActual,3);            
            
            % State after Step 2
            step(testCase.designer,options,[]);
            stateActual = get(testCase.designer,'IsPreviousStepFixed');            
            countActual = get(testCase.designer,'Count');            
            testCase.verifyTrue(stateActual);
            testCase.verifyEqual(countActual,4);                        

        end        
        
        % Test 
        function testNsoltDictionaryLearningIhtDec112Ch22Ord222(testCase)
    
            % Parameter settings
            nCoefs   = 4;
            nLevels  = 1;
            nDecs    = [ 1 1 2 ];
            nChs     = [ 2 2 ];
            nOrds    = [ 2 2 2 ];
            srcImgs{1} = rand(16,16,16);
            optfcn = @fminunc;
            nUnfixedSteps = 2;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'IsFixedCoefs',true,...
                'NumberOfUnfixedInitialSteps',nUnfixedSteps,...
                'NumberOfDimensions','Three',...
                'TrainingImages',srcImgs,...
                'SparseApproximation','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds,...
                'DecimationFactor',nDecs);

            % Options
            options = optimoptions(optfcn);
            options = optimoptions(options,'Algorithm','quasi-newton');
            options = optimoptions(options,'Display','off');
            options = optimoptions(options,'MaxIter',2);
            
            % State after Step 1
            step(testCase.designer,options,[]);
            stateActual = get(testCase.designer,'IsPreviousStepFixed');
            countActual = get(testCase.designer,'Count');            
            testCase.verifyFalse(stateActual);
            testCase.verifyEqual(countActual,2);
                        
            % State after Step 2
            step(testCase.designer,options,[]);
            stateActual = get(testCase.designer,'IsPreviousStepFixed');            
            countActual = get(testCase.designer,'Count');            
            testCase.verifyFalse(stateActual);
            testCase.verifyEqual(countActual,3);            
            
            % State after Step 2
            step(testCase.designer,options,[]);
            stateActual = get(testCase.designer,'IsPreviousStepFixed');            
            countActual = get(testCase.designer,'Count');            
            testCase.verifyTrue(stateActual);
            testCase.verifyEqual(countActual,4);                        

        end        
        
    end
end

