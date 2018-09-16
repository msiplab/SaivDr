classdef NsoltDictionaryLearningPnPTestCase < matlab.unittest.TestCase
    %NSOLTDICTIONARYLEARNINGPNPTESTCASE Test case for NsoltDictionaryLearningPnP
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2018, Shogo MURAMATSU
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
       target
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.target);
        end
    end
    
    methods (Test)
        
        function testConstruction(testCase)
            
            % Expected value
            trnImgsExpctd = [];
            sprsAprxExpctd = [];
            dicUpdExpctd = [];
            cntExpctd = [];
            ndecExpctd = [2 2];
            nchsExpctd = [2 2];
            nordExpctd = [0 0];
            nlvExpctd = 1;
            nvmExpctd = 0;
            dtypeExpctd = 'Image';
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.target = NsoltDictionaryLearningPnP();
            
            % Actual value
            trnImgsActual = testCase.target.TrainingImages;
            sprsAprxActual = testCase.target.SparseApproximater;
            dicUpdActual = testCase.target.DictionaryUpdater;            
            cntActual = testCase.target.Count;
            ndecActual = testCase.target.DecimationFactor;
            nchsActual = testCase.target.NumberOfChannels;
            nordActual = testCase.target.PolyPhaseOrder;
            nvmActual = testCase.target.NumberOfVanishingMoments;
            nlvActual = testCase.target.NumberOfLevels;
            dtypeActual = testCase.target.DataType;

            % Evaluation
            testCase.verifyEqual(trnImgsActual,trnImgsExpctd);
            testCase.verifyEqual(sprsAprxActual,sprsAprxExpctd);
            testCase.verifyEqual(dicUpdActual,dicUpdExpctd);
            testCase.verifyEqual(cntActual,cntExpctd);            
            testCase.verifyEqual(ndecActual,ndecExpctd);            
            testCase.verifyEqual(nchsActual,nchsExpctd);            
            testCase.verifyEqual(nordActual,nordExpctd);            
            testCase.verifyEqual(nvmActual,nvmExpctd);
            testCase.verifyEqual(nlvActual,nlvExpctd);            
            testCase.verifyEqual(dtypeActual,dtypeExpctd);            
            
        end
        
        function testSetPropertiesImage(testCase)
            
            % Configuration
            nDecs = [2 2];
            nChs  = [4 4];
            nOrd  = [2 2];
            nVms  = 0;

            % Expected value
            import saivdr.sparserep.*
            sprsAprxExpctd = IterativeHardThresholding();
            import saivdr.dictionary.nsoltx.design.*
            dicUpdExpctd =  NsoltDictionaryUpdateSgd();
            import saivdr.dictionary.nsoltx.*
            olppufbExpctd = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',nOrd,...
                'NumberOfVanishingMoments',nVms,...
                'OutputMode','ParameterMatrixSet');

            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.target = NsoltDictionaryLearningPnP(...
                'SparseApproximater',sprsAprxExpctd,...
                'DictionaryUpdater', dicUpdExpctd,...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'NumberOfVanishingMoments',nVms,...                
                'PolyPhaseOrder',nOrd);
            
            % Actual value
            sprsAprxActual = testCase.target.SparseApproximater;
            dicUpdActual = testCase.target.DictionaryUpdater;     
            olppufbActual = testCase.target.OvsdLpPuFb;

            % Evaluation
            testCase.verifyClass(sprsAprxActual,class(sprsAprxExpctd));
            testCase.verifyClass(dicUpdActual,class(dicUpdExpctd));
            testCase.verifyEqual(olppufbActual,olppufbExpctd);
        end
        
        function testStepImage(testCase)
            
            % Configuration
            nDecs = [2 2];
            nChs  = [4 4];
            nOrd  = [2 2];
            nVms  = 0;

            % Parameters
            nSprsCoefs = 4;
            isOptMus = false;
            srcImgs{1} = rand(16,16);                     

            % Expected value
            import saivdr.sparserep.*
            sprsAprx = IterativeHardThresholding(...
                'NumberOfSparseCoefficients',nSprsCoefs);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateSgd(...
                'GradObj','on');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.target = NsoltDictionaryLearningPnP(...
                'SparseApproximater',sprsAprx,...
                'DictionaryUpdater', dicUpd,...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'NumberOfVanishingMoments',nVms,...                
                'PolyPhaseOrder',nOrd);
            
            % Preparation
            lppufbPre = testCase.target.OvsdLpPuFb;
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysis2dSystem(lppufbPre);
            import saivdr.sparserep.*
            sprsAprxClone = sprsAprx.clone();
            sprsAprxClone.Dictionary{1} = synthesizer;
            sprsAprxClone.Dictionary{2} = analyzer;
            [~, coefsPre{1},scales{1}] = sprsAprxClone.step(srcImgs{1});
            aprxErr = AprxErrorWithSparseRep(...
                'TrainingImages', srcImgs);
            costPre = aprxErr.step(lppufbPre,coefsPre,scales);
            
            % Pst
            options = optimoptions('fminunc');
            options = optimoptions(options,'Algorithm','quasi-newton');
            options = optimoptions(options,'Display','off');
            options = optimoptions(options,'MaxIter',2);
            [~, costPst] = testCase.target.step(srcImgs,options);

            % Evaluation
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre));

        end
        
        function testSetPropertiesVolumetricData(testCase)
            
            % Configuration
            nDecs = [2 2 2];
            nChs  = [6 6];
            nOrd  = [2 2 2];
            nVms  = 0;   
            
            % Expected value
            import saivdr.sparserep.*
            sprsAprxExpctd = IterativeHardThresholding();
            import saivdr.dictionary.nsoltx.design.*
            dicUpdExpctd =  NsoltDictionaryUpdateSgd();
            import saivdr.dictionary.nsoltx.*
            olppufbExpctd = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',nOrd,...
                'NumberOfVanishingMoments',nVms,...
                'OutputMode','ParameterMatrixSet');

            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.target = NsoltDictionaryLearningPnP(...
                'DataType','Volumetric Data',...
                'SparseApproximater',sprsAprxExpctd,...
                'DictionaryUpdater', dicUpdExpctd,...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'NumberOfVanishingMoments',nVms,...                
                'PolyPhaseOrder',nOrd);
            
            % Actual value
            sprsAprxActual = testCase.target.SparseApproximater;
            dicUpdActual = testCase.target.DictionaryUpdater;     
            olppufbActual = testCase.target.OvsdLpPuFb;

            % Evaluation
            testCase.verifyClass(sprsAprxActual,class(sprsAprxExpctd));
            testCase.verifyClass(dicUpdActual,class(dicUpdExpctd));
            testCase.verifyEqual(olppufbActual,olppufbExpctd);
            
        end
            
        %{
            
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
...                'NumberOfSparseCoefficients',nCoefs,...
            
            testCase.designer = NsoltDictionaryLearningPnP(...
                'TrainingImages',srcImgs,...
                'SparseApproximation',sprsaprx,...
                'DictionaryUpdate',dicupd,...
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
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
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
        function testNsoltDictionaryLearningPnPGpDec22Ch62Ord44(testCase)
    
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
            testCase.designer = NsoltDictionaryLearningPnP(...
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
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
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
        function testNsoltDictionaryLearningPnPGpDec22Ch62Ord44Ga(testCase)
    
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
            testCase.designer = NsoltDictionaryLearningPnP(...
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
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
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
        function testNsoltDictionaryLearningPnPIhtDec22Ch62Ord44(testCase)
            
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
            testCase.designer = NsoltDictionaryLearningPnP(...
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
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
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
        function testNsoltDictionaryLearningPnPIhtDec22Ch44Ord44Sgd(testCase)
            
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
            testCase.designer = NsoltDictionaryLearningPnP(...
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
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
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
        function testNsoltDictionaryLearningPnPIhtDec22Ch44Ord44GradObj(testCase)
            
            % Parameter settings
            nLevels = 1;
            nChs  = [ 4 4 ];
            nOrds = [ 4 4 ];
            nSprsCoefs = 4;
            isOptMus = false;
            srcImgs{1} = rand(12,16);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.design.*
            testCase.designer = NsoltDictionaryLearningPnP(...
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
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
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
        function testNsoltDictionaryLearningPnPIhtDec22Ch62Ord44Ga(testCase)
    
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
            testCase.designer = NsoltDictionaryLearningPnP(...
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
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
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
        function testNsoltDictionaryLearningPnPGpDec222Ch55Ord222(testCase)
            
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
            testCase.designer = NsoltDictionaryLearningPnP(...
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
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
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
        function testNsoltDictionaryLearningPnPGpDec222Ch64Ord222Ga(testCase)
    
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
            testCase.designer = NsoltDictionaryLearningPnP(...
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
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
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
        function testNsoltDictionaryLearningPnPIhtDec222Ch64Ord222(testCase)
    
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
            testCase.designer = NsoltDictionaryLearningPnP(...
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
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
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
        function testNsoltDictionaryLearningPnPIhtDec222Ch55Ord222Ga(testCase)
    
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
            testCase.designer = NsoltDictionaryLearningPnP(...
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
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
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
        
        % Test
        function testNsoltDictionaryLearningPnPIhtDec22Ch44Ord22(testCase)
    
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
            testCase.designer = NsoltDictionaryLearningPnP(...
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
        function testNsoltDictionaryLearningPnPIhtDec222Ch55Ord222(testCase)
    
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
            testCase.designer = NsoltDictionaryLearningPnP(...
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
        function testNsoltDictionaryLearningPnPIhtDec112Ch22Ord222(testCase)
    
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
            testCase.designer = NsoltDictionaryLearningPnP(...
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
        %}
    end
end
