classdef NsoltDictionaryLearningTestCase < matlab.unittest.TestCase
    %NSOLTDICTIONARYLEARNINGTESTCASE Test case for NsoltDictionaryLearning
    %
    % SVN identifier:
    % $Id: NsoltDictionaryLearningTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        designer;
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
            import saivdr.dictionary.nsolt.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'SourceImages',srcImgs,...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'NumbersOfPolyphaseOrder',nOrds);
            
            % Pre
            lppufbPre = get(testCase.designer,'LpPuFb2d');
            import saivdr.dictionary.nsolt.*
            synthesizer = NsoltFactory.createSynthesisSystem(lppufbPre);
            analyzer  = NsoltFactory.createAnalysisSystem(lppufbPre);
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
            import saivdr.dictionary.nsolt.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'SourceImages',srcImgs,...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2);
            
            % Pre
            import saivdr.sparserep.*
            lppufbPre = get(testCase.designer,'LpPuFb2d');
            import saivdr.dictionary.nsolt.*
            synthesizer = NsoltFactory.createSynthesisSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysisSystem(lppufbPre);            
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
            import matlab.unittest.constraints.IsLessThan;
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
            import saivdr.dictionary.nsolt.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'SourceImages',srcImgs,...
                'SparseCoding','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'NumbersOfPolyphaseOrder',nOrds);
            
            % Pre
            lppufbPre = get(testCase.designer,'LpPuFb2d');
            import saivdr.dictionary.nsolt.*
            synthesizer = NsoltFactory.createSynthesisSystem(lppufbPre);
            analyzer  = NsoltFactory.createAnalysisSystem(lppufbPre);
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
            import matlab.unittest.constraints.IsLessThan;
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
            import saivdr.dictionary.nsolt.design.*
            testCase.designer = NsoltDictionaryLearning(...
                'SourceImages',srcImgs,...
                'SparseCoding','IterativeHardThresholding',...
                'NumberOfSparseCoefficients',nCoefs,...
                'NumberOfTreeLevels',nLevels,...
                'NumberOfSymmetricChannel',nChs(1),...
                'NumberOfAntisymmetricChannel',nChs(2),...
                'OptimizationFunction',optfcn,...                
                'NumbersOfPolyphaseOrder',nOrds,...
                'MaxIterOfHybridFmin',2,...
                'GenerationFactorForMus',2);
            
            % Pre
            import saivdr.sparserep.*
            lppufbPre = get(testCase.designer,'LpPuFb2d');
            import saivdr.dictionary.nsolt.*
            synthesizer = NsoltFactory.createSynthesisSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysisSystem(lppufbPre);            
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
            import matlab.unittest.constraints.IsLessThan;
            testCase.verifyThat(costPst, IsLessThan(costPre));            

        end
                
        
    end
end
