classdef CnsoltDictionaryLearningTestCase < matlab.unittest.TestCase
    %CnsoltDictionaryLEARNINGTESTCASE Test case for CnsoltDictionaryLearning
    %
    % SVN identifier:
    % $Id: CnsoltDictionaryLearningTestCase.m 866 2015-11-24 04:29:42Z sho $
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627    
    %
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
        function testCnsoltDictionaryLearningGpDec22Ch7Ord44(testCase)
    
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = 7;
            nOrds = [ 4 4 ];
            nSprsCoefs = 4;
            isOptMus = false;
            srcImgs{1} = rand(16,16).*exp(1i*2*pi*rand(16,16));            
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.design.*
            testCase.designer = CnsoltDictionaryLearning(...
                'SourceImages',srcImgs,...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfChannels',nChs,...
                'NumbersOfPolyphaseOrder',nOrds);
            
            % Pre
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.cnsoltx.*
            synthesizer = CnsoltFactory.createSynthesis2dSystem(lppufbPre);
            analyzer  = CnsoltFactory.createAnalysis2dSystem(lppufbPre);
            import saivdr.sparserep.*
            gpnsolt = GradientPursuit(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'NumberOfTreeLevels',nLevels);
            [~, coefsPre{1},scales{1}] = step(gpnsolt,srcImgs{1},nSprsCoefs);
            aprxErr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels);
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
        function testCnsoltDictionaryLearningGpDec22Ch7Ord44Ga(testCase)
    
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = 7;
            nOrds = [ 4 4 ];
            nSprsCoefs = 4;
            isOptMus = true;
            srcImgs{1} = rand(16,16).*exp(1i*2*pi*rand(16,16));            
            optfcn = @ga;
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.design.*
            testCase.designer = CnsoltDictionaryLearning(...
                'SourceImages',srcImgs,...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfChannels',nChs,...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2);
            
            % Pre
            import saivdr.sparserep.*
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.cnsoltx.*
            synthesizer = CnsoltFactory.createSynthesis2dSystem(lppufbPre);
            analyzer    = CnsoltFactory.createAnalysis2dSystem(lppufbPre);            
            gpnsolt = GradientPursuit(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'NumberOfTreeLevels',nLevels);
            [~, coefsPre{1}, scales{1}] = step(gpnsolt,srcImgs{1},nSprsCoefs);
            aprxErr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels);
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
        function testCnsoltDictionaryLearningIhtDec22Ch7Ord44(testCase)
            
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = 7;
            nOrds = [ 4 4 ];
            nSprsCoefs = 4;
            isOptMus = false;
            srcImgs{1} = rand(16,16).*exp(1i*2*pi*rand(16,16));
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.design.*
            testCase.designer = CnsoltDictionaryLearning(...
                'SourceImages',srcImgs,...
                'SparseCoding','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfChannels',nChs,...
                'NumbersOfPolyphaseOrder',nOrds);
            
            % Pre
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.cnsoltx.*
            synthesizer = CnsoltFactory.createSynthesis2dSystem(lppufbPre);
            analyzer  = CnsoltFactory.createAnalysis2dSystem(lppufbPre);
            import saivdr.sparserep.*
            gpnsolt = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'NumberOfTreeLevels',nLevels);
            [~, coefsPre{1},scales{1}] = step(gpnsolt,srcImgs{1},nSprsCoefs);
            aprxErr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels);
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
%         function testCnsoltDictionaryLearningIhtDec22Ch8Ord44Sgd(testCase)
%             
%             % Parameter settings
%             nCoefs = 4;
%             nLevels = 1;
%             nChs  = 8;
%             nOrds = [ 4 4 ]; 
%             nSprsCoefs = 4;
%             isOptMus = false;
%             srcImgs{1} = rand(12,16).*exp(1i*2*pi*rand(12,16));
%             srcImgs{2} = rand(12,16).*exp(1i*2*pi*rand(12,16));
%             srcImgs{3} = rand(12,16).*exp(1i*2*pi*rand(12,16));
%             srcImgs{4} = rand(12,16).*exp(1i*2*pi*rand(12,16));
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.design.*
%             testCase.designer = CnsoltDictionaryLearning(...
%                 'SourceImages',srcImgs,...
%                 'SparseCoding','IterativeHardThresholding',...
%                 'DictionaryUpdater','CnsoltDictionaryUpdateSgd',...
%                 'IsFixedCoefs',true,...
%                 'NumberOfSparseCoefficients',nCoefs,...
%                 'NumberOfTreeLevels',nLevels,...
%                 'NumberOfChannels',nChs,...
%                 'NumbersOfPolyphaseOrder',nOrds,...
%                 'GradObj', 'on');
%             
%             % Pre
%             lppufbPre = get(testCase.designer,'OvsdLpPuFb');
%             import saivdr.dictionary.cnsoltx.*
%             synthesizer = CnsoltFactory.createSynthesis2dSystem(lppufbPre);
%             analyzer  = CnsoltFactory.createAnalysis2dSystem(lppufbPre);
%             import saivdr.sparserep.*
%             ihtnsolt = IterativeHardThresholding(...
%                 'Synthesizer',synthesizer,...
%                 'AdjOfSynthesizer',analyzer,...
%                 'NumberOfTreeLevels',nLevels);
%             nImgs = length(srcImgs);
%             coefsPre = cell(nImgs,1);
%             setOfScales   = cell(nImgs,1);
%             for iImg = 1:nImgs
%                 [~, coefsPre{iImg},setOfScales{iImg}] = ...
%                     step(ihtnsolt,srcImgs{iImg},nSprsCoefs);
%             end
%             aprxErr = AprxErrorWithSparseRep(...
%                 'SourceImages', srcImgs,...
%                 'NumberOfTreeLevels',nLevels);
%             costPre = step(aprxErr,lppufbPre,coefsPre,setOfScales);
%             
%             % Pst
%             options = optimset(...
%                 'MaxIter',2*nImgs,...
%                 'TolX',1e-4);                
%             [~, costPst] = step(testCase.designer,options,isOptMus);
%             
%             % Evaluation
%             import matlab.unittest.constraints.IsLessThan
%             testCase.verifyThat(costPst, IsLessThan(costPre));
%             
%         end
%         
%         % Test
%         function testCnsoltDictionaryLearningIhtDec22Ch8Ord44GradObj(testCase)
%             
%             % Parameter settings
%             nLevels = 1;
%             nChs  = 8;
%             nOrds = [ 4 4 ];
%             nSprsCoefs = 4;
%             isOptMus = false;
%             srcImgs{1} = rand(12,16).*exp(1i*2*pi*rand(12,16));
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.design.*
%             testCase.designer = CnsoltDictionaryLearning(...
%                 'SourceImages',srcImgs,...
%                 'SparseCoding','IterativeHardThresholding',...
%                 'NumberOfSparseCoefficients',nSprsCoefs,...
%                 'NumberOfTreeLevels',nLevels,...
%                 'NumberOfChannels',nChs,...
%                 'NumbersOfPolyphaseOrder',nOrds,...
%                 'IsFixedCoefs',true,...
%                 'GradObj', 'on');
%             
%             % Pre
%             lppufbPre = get(testCase.designer,'OvsdLpPuFb');
%             import saivdr.dictionary.cnsoltx.*
%             synthesizer = CnsoltFactory.createSynthesis2dSystem(lppufbPre);
%             analyzer  = CnsoltFactory.createAnalysis2dSystem(lppufbPre);
%             import saivdr.sparserep.*
%             gpnsolt = IterativeHardThresholding(...
%                 'Synthesizer',synthesizer,...
%                 'AdjOfSynthesizer',analyzer,...
%                 'NumberOfTreeLevels',nLevels);
%             [~, coefsPre{1},scales{1}] = step(gpnsolt,srcImgs{1},nSprsCoefs);
%             aprxErr = AprxErrorWithSparseRep(...
%                 'SourceImages', srcImgs,...
%                 'NumberOfTreeLevels',nLevels);
%             costPre = step(aprxErr,lppufbPre,coefsPre,scales);
%             
%             % Pst
%             options = optimoptions('fminunc');
%             options = optimoptions(options,'Algorithm','trust-region');
%             options = optimoptions(options,'Display','off');
%             options = optimoptions(options,'MaxIter',2);
%             options = optimoptions(options,'GradObj','on');
%             [~, costPst] = step(testCase.designer,options,isOptMus);
%             
%             % Evaluation
%             import matlab.unittest.constraints.IsLessThan
%             testCase.verifyThat(costPst, IsLessThan(costPre));
%             
%         end
        
        % Test 
        function testCnsoltDictionaryLearningIhtDec22Ch7Ord44Ga(testCase)
    
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = 7;
            nOrds = [ 4 4 ];
            nSprsCoefs = 4;
            isOptMus = true;
            srcImgs{1} = rand(16,16).*exp(1i*2*pi*rand(16,16));            
            optfcn = @ga;
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.design.*
            testCase.designer = CnsoltDictionaryLearning(...
                'SourceImages',srcImgs,...
                'SparseCoding','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfChannels',nChs,...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2);
            
            % Pre
            import saivdr.sparserep.*
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.cnsoltx.*
            synthesizer = CnsoltFactory.createSynthesis2dSystem(lppufbPre);
            analyzer    = CnsoltFactory.createAnalysis2dSystem(lppufbPre);            
            gpnsolt = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'NumberOfTreeLevels',nLevels);
            [~, coefsPre{1}, scales{1}] = step(gpnsolt,srcImgs{1},nSprsCoefs);
            aprxErr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels);
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
        function testCnsoltDictionaryLearningGpDec222Ch10Ord222(testCase)
            
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = 10;
            nOrds = [ 2 2 2 ];
            nSprsCoefs = 4;
            isOptMus = false;
            srcImgs{1} = rand(16,16,16).*exp(1i*2*pi*rand(16,16,16));
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.design.*
            testCase.designer = CnsoltDictionaryLearning(...
                'NumberOfDimensions','Three',...
                'SourceImages',srcImgs,...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfChannels',nChs,...
                'NumbersOfPolyphaseOrder',nOrds);
            
            % Pre
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.cnsoltx.*
            synthesizer = CnsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer  = CnsoltFactory.createAnalysis3dSystem(lppufbPre);
            import saivdr.sparserep.*
            gpnsolt = GradientPursuit(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'NumberOfTreeLevels',nLevels);
            [~, coefsPre{1},scales{1}] = step(gpnsolt,srcImgs{1},nSprsCoefs);
            aprxErr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels);
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
        function testCnsoltDictionaryLearningGpDec222Ch9Ord222Ga(testCase)
    
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = 9;
            nOrds = [ 2 2 2 ];
            nSprsCoefs = 4;
            isOptMus = true;
            srcImgs{1} = rand(16,16,16).*exp(1i*2*pi*rand(16,16,16));            
            optfcn = @ga;
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.design.*
            testCase.designer = CnsoltDictionaryLearning(...
                'NumberOfDimensions','Three',...
                'SourceImages',srcImgs,...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfChannels',nChs,...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2);
            
            % Pre
            import saivdr.sparserep.*
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.cnsoltx.*
            synthesizer = CnsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer    = CnsoltFactory.createAnalysis3dSystem(lppufbPre);            
            gpnsolt = GradientPursuit(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'NumberOfTreeLevels',nLevels);
            [~, coefsPre{1}, scales{1}] = step(gpnsolt,srcImgs{1},nSprsCoefs);
            aprxErr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels);
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
        function testCnsoltDictionaryLearningIhtDec222Ch9Ord222(testCase)
    
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = 9;
            nOrds = [ 2 2 2 ];
            nSprsCoefs = 4;
            isOptMus = false;
            srcImgs{1} = rand(16,16,16).*exp(1i*2*pi*rand(16,16,16));            
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.design.*
            testCase.designer = CnsoltDictionaryLearning(...
                'NumberOfDimensions','Three',...
                'SourceImages',srcImgs,...
                'SparseCoding','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfChannels',nChs,...
                'NumbersOfPolyphaseOrder',nOrds);
            
            % Pre
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.cnsoltx.*
            synthesizer = CnsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer  = CnsoltFactory.createAnalysis3dSystem(lppufbPre);
            import saivdr.sparserep.*
            gpnsolt = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'NumberOfTreeLevels',nLevels);
            [~, coefsPre{1},scales{1}] = step(gpnsolt,srcImgs{1},nSprsCoefs);
            aprxErr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels);
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
        function testCnsoltDictionaryLearningIhtDec222Ch10Ord222Ga(testCase)
    
            % Parameter settings
            nCoefs = 4;
            nLevels = 1;
            nChs = 10;
            nOrds = [ 2 2 2 ];
            nSprsCoefs = 4;
            isOptMus = true;
            srcImgs{1} = rand(16,16,16).*exp(1i*2*pi*rand(16,16,16));            
            optfcn = @ga;
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.design.*
            testCase.designer = CnsoltDictionaryLearning(...
                'NumberOfDimensions','Three',...
                'SourceImages',srcImgs,...
                'SparseCoding','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfChannels',nChs,...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2);
            
            % Pre
            import saivdr.sparserep.*
            lppufbPre = get(testCase.designer,'OvsdLpPuFb');
            import saivdr.dictionary.cnsoltx.*
            synthesizer = CnsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer    = CnsoltFactory.createAnalysis3dSystem(lppufbPre);            
            gpnsolt = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'NumberOfTreeLevels',nLevels);
            [~, coefsPre{1}, scales{1}] = step(gpnsolt,srcImgs{1},nSprsCoefs);
            aprxErr = AprxErrorWithSparseRep(...
                'SourceImages', srcImgs,...
                'NumberOfTreeLevels',nLevels);
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
        
        % Test 
        function testCnsoltDictionaryLearningIhtDec22Ch8Ord22(testCase)
    
            % Parameter settings
            nCoefs   = 4;
            nLevels  = 1;
            nChs     = 8;
            nOrds    = [ 2 2 ];
            srcImgs{1} = rand(16,16).*exp(1i*2*pi*rand(16,16));
            optfcn = @fminunc;
            nUnfixedSteps = 1;
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.design.*
            testCase.designer = CnsoltDictionaryLearning(...
                'IsFixedCoefs',true,...
                'NumberOfUnfixedInitialSteps', nUnfixedSteps,...
                'NumberOfDimensions','Two',...
                'SourceImages',srcImgs,...
                'SparseCoding','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfChannels',nChs,...
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
        function testCnsoltDictionaryLearningIhtDec222Ch10Ord222(testCase)
    
            % Parameter settings
            nCoefs   = 4;
            nLevels  = 1;
            nChs     = 10;
            nOrds    = [ 2 2 2 ];
            srcImgs{1} = rand(16,16,16).*exp(1i*2*pi*rand(16,16,16));
            optfcn = @fminunc;
            nUnfixedSteps = 2;
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.design.*
            testCase.designer = CnsoltDictionaryLearning(...
                'IsFixedCoefs',true,...
                'NumberOfUnfixedInitialSteps',nUnfixedSteps,...
                'NumberOfDimensions','Three',...
                'SourceImages',srcImgs,...
                'SparseCoding','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfChannels',nChs,...
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
        function testCnsoltDictionaryLearningIhtDec112Ch4Ord222(testCase)
    
            % Parameter settings
            nCoefs   = 4;
            nLevels  = 1;
            nDecs    = [ 1 1 2 ];
            nChs     = 4;
            nOrds    = [ 2 2 2 ];
            srcImgs{1} = rand(16,16,16).*exp(1i*2*pi*rand(16,16,16));
            optfcn = @fminunc;
            nUnfixedSteps = 2;
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.design.*
            testCase.designer = CnsoltDictionaryLearning(...
                'IsFixedCoefs',true,...
                'NumberOfUnfixedInitialSteps',nUnfixedSteps,...
                'NumberOfDimensions','Three',...
                'SourceImages',srcImgs,...
                'SparseCoding','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfChannels',nChs,...
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
