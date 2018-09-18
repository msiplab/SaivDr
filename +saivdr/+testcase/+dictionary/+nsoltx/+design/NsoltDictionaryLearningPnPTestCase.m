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
    properties (TestParameter)
        %useparallel = struct('true', true, 'false', false );
        %usegpu = struct('true', true, 'false', false );
        vdec = struct('small',1,'medium',2);
        hdec = struct('small',1,'medium',2);
        sch = struct('small',4,'medium',6);
        ach = struct('small',4,'medium',6);        
        ddec = struct('small',1,'medium',2);
        height = struct('small',8,'large',32);
        width = struct('small',8,'large',32);
        depth = struct('small',8,'large',32);
        isfista = struct('true', true, 'false', false );
    end
    
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
        
        function testStepIhtImage(testCase)
            
            % Configuration
            nDecs = [2 2];
            nChs  = [4 4];
            nOrd  = [2 2];
            nVms  = 0;
            
            % Parameters
            nSprsCoefs = 4;
            srcImgs{1} = rand(16,16);
            
            % Expected value
            import saivdr.sparserep.*
            sprsAprx = IterativeHardThresholding(...
                'NumberOfSparseCoefficients',nSprsCoefs);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateSgd(...
                'GradObj','on',...
                'Step','AdaGrad');
            
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
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre*1.01));
            
        end
        
        function testSetPropertiesIhtVolumetricData(testCase)
            
            % Configuration
            nDecs = [2 2 2];
            nChs  = [6 6];
            nOrd  = [2 2 2];
            nVms  = 0;
            
            % Parameters
            nSprsCoefs = 4;
            srcImgs{1} = rand(16,16,16);
            
            % Expected value
            import saivdr.sparserep.*
            sprsAprx = IterativeHardThresholding(...
                'NumberOfSparseCoefficients',nSprsCoefs);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateSgd(...
                'GradObj','on',...
                'Step','AdaGrad');
            import saivdr.dictionary.nsoltx.*
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.target = NsoltDictionaryLearningPnP(...
                'DataType','Volumetric Data',...
                'SparseApproximater',sprsAprx,...
                'DictionaryUpdater', dicUpd,...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'NumberOfVanishingMoments',nVms,...
                'PolyPhaseOrder',nOrd);
            
            % Preparation
            lppufbPre = testCase.target.OvsdLpPuFb;
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysis3dSystem(lppufbPre);
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
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre*1.01));
            
        end
        
        function testStepGpImage(testCase)
            
            % Configuration
            nDecs = [2 2];
            nChs  = [4 4];
            nOrd  = [2 2];
            nVms  = 0;
            
            % Parameters
            nSprsCoefs = 4;
            srcImgs{1} = rand(16,16);
            
            % Expected value
            import saivdr.sparserep.*
            sprsAprx = GradientPursuit(...
                'NumberOfSparseCoefficients',nSprsCoefs);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateSgd(...
                'GradObj','on',...
                'Step','AdaGrad');
            
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
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre*1.01));
            
        end
        
        function testStepGpVolumetricData(testCase)
            
            % Configuration
            nDecs = [2 2 2];
            nChs  = [6 6];
            nOrd  = [2 2 2];
            nVms  = 0;
            
            % Parameters
            nSprsCoefs = 4;
            srcImgs{1} = rand(16,16, 16);
            
            % Expected value
            import saivdr.sparserep.*
            sprsAprx = GradientPursuit(...
                'NumberOfSparseCoefficients',nSprsCoefs);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateSgd(...
                'GradObj','on',...
                'Step','AdaGrad');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.target = NsoltDictionaryLearningPnP(...
                'DataType','Volumetric Data',...
                'SparseApproximater',sprsAprx,...
                'DictionaryUpdater', dicUpd,...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'NumberOfVanishingMoments',nVms,...
                'PolyPhaseOrder',nOrd);
            
            % Preparation
            lppufbPre = testCase.target.OvsdLpPuFb;
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysis3dSystem(lppufbPre);
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
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre*1.01));
            
        end
        
        function testStepIstaImage(testCase,isfista)
            
            % Configuration
            nDecs = [2 2];
            nChs  = [4 4];
            nOrd  = [2 2];
            nVms  = 1;
            
            % Parameters
            lambda = 1e-3;
            srcImgs{1} = randn(16,16)+0.5;
            
            % Expected value
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            if isfista
                algorithm = IstaSystem();
            else
                algorithm = FistaSystem();
            end
            algorithm.Lambda = lambda;
            sprsAprx = IterativeSparseApproximater(...
                'Algorithm', algorithm,...
                'MaxIter',4);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateSgd(...
                'GradObj','on',...
                'Step','AdaGrad');
            
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
            options = optimoptions(options,'MaxIter',4);
            [~, costPst] = testCase.target.step(srcImgs,options);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre*1.01));
            
        end
        
        function testStepIstaVolumetricData(testCase,isfista)
            
            % Configuration
            nDecs = [2 2 2];
            nChs  = [6 6];
            nOrd  = [2 2 2];
            nVms  = 1;
            
            % Parameters
            lambda = 1e-3;
            srcImgs{1} = randn(16,16,16)+0.5;
            
            % Expected value
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            if isfista
                algorithm = IstaSystem();
            else
                algorithm = FistaSystem();
            end
            algorithm.Lambda = lambda;
            sprsAprx = IterativeSparseApproximater(...
                'Algorithm', algorithm,...
                'MaxIter',4);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateSgd(...
                'GradObj','on',...
                'Step','AdaGrad');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.target = NsoltDictionaryLearningPnP(...
                'DataType','Volumetric Data',...
                'SparseApproximater',sprsAprx,...
                'DictionaryUpdater', dicUpd,...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'NumberOfVanishingMoments',nVms,...
                'PolyPhaseOrder',nOrd);
            
            % Preparation
            lppufbPre = testCase.target.OvsdLpPuFb;
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysis3dSystem(lppufbPre);
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
            options = optimoptions(options,'MaxIter',4);
            [~, costPst] = testCase.target.step(srcImgs,options);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre*1.01));
            
        end
        
        function testStepIstaImageSize(testCase,height,width)
            
            % Configuration
            nDecs = [2 2];
            nChs  = [4 4];
            nOrd  = [2 2];
            nVms  = 1;
            
            % Parameters
            lambda = 1e-3;
            srcImgs{1} = randn(height,width)+0.5;
            
            % Expected value
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            algorithm = FistaSystem();
            algorithm.Lambda = lambda;
            sprsAprx = IterativeSparseApproximater(...
                'Algorithm', algorithm,...
                'MaxIter',4);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateSgd(...
                'GradObj','on',...
                'Step','AdaGrad');
            
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
            options = optimoptions(options,'MaxIter',4);
            [~, costPst] = testCase.target.step(srcImgs,options);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre*1.01));
            
        end
        
        function testStepIstaVolumetricDataSize(testCase,...
                height,width,depth)
            
            % Configuration
            nDecs = [2 2 2];
            nChs  = [6 6];
            nOrd  = [2 2 2];
            nVms  = 1;
            
            % Parameters
            lambda = 1e-3;
            srcImgs{1} = randn(height,width,depth)+0.5;
            
            % Expected value
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            algorithm = FistaSystem();
            algorithm.Lambda = lambda;
            sprsAprx = IterativeSparseApproximater(...
                'Algorithm', algorithm,...
                'MaxIter',4);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateSgd(...
                'GradObj','on',...
                'Step','AdaGrad');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.target = NsoltDictionaryLearningPnP(...
                'DataType','Volumetric Data',...
                'SparseApproximater',sprsAprx,...
                'DictionaryUpdater', dicUpd,...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'NumberOfVanishingMoments',nVms,...
                'PolyPhaseOrder',nOrd);
            
            % Preparation
            lppufbPre = testCase.target.OvsdLpPuFb;
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysis3dSystem(lppufbPre);
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
            options = optimoptions(options,'MaxIter',4);
            [~, costPst] = testCase.target.step(srcImgs,options);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre*1.01));
        end
        
        
        function testStepIstaImageDecimation(testCase,vdec,hdec)
            
            % Configuration
            nDecs = [vdec hdec];
            nChs  = [4 4];
            nOrd  = [2 2];
            nVms  = 1;
            
            % Parameters
            lambda = 1e-3;
            srcImgs{1} = randn(16,16)+0.5;
            
            % Expected value
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            algorithm = FistaSystem();
            algorithm.Lambda = lambda;
            sprsAprx = IterativeSparseApproximater(...
                'Algorithm', algorithm,...
                'MaxIter',4);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateSgd(...
                'GradObj','on',...
                'Step','AdaGrad');
            
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
            options = optimoptions(options,'MaxIter',4);
            [~, costPst] = testCase.target.step(srcImgs,options);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre*1.01));
            
        end
        
        function testStepIstaVolumetricDataDecimation(testCase,...
                vdec,hdec,ddec)
            
            % Configuration
            nDecs = [vdec hdec ddec];
            nChs  = [6 6];
            nOrd  = [2 2 2];
            nVms  = 1;
            
            % Parameters
            lambda = 1e-3;
            srcImgs{1} = randn(16,16,16)+0.5;
            
            % Expected value
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            algorithm = FistaSystem();
            algorithm.Lambda = lambda;
            sprsAprx = IterativeSparseApproximater(...
                'Algorithm', algorithm,...
                'MaxIter',4);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateSgd(...
                'GradObj','on',...
                'Step','AdaGrad');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.target = NsoltDictionaryLearningPnP(...
                'DataType','Volumetric Data',...
                'SparseApproximater',sprsAprx,...
                'DictionaryUpdater', dicUpd,...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'NumberOfVanishingMoments',nVms,...
                'PolyPhaseOrder',nOrd);
            
            % Preparation
            lppufbPre = testCase.target.OvsdLpPuFb;
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysis3dSystem(lppufbPre);
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
            options = optimoptions(options,'MaxIter',4);
            [~, costPst] = testCase.target.step(srcImgs,options);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre*1.01));
            
        end
        
        
        function testStepIstaImageChannel(testCase,sch,ach)
            
            % Configuration
            nDecs = [2 2];
            nChs  = [sch ach];
            nOrd  = [2 2];
            nVms  = 1;
            
            % Parameters
            lambda = 1e-3;
            srcImgs{1} = randn(16,16)+0.5;
            
            % Expected value
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            algorithm = FistaSystem();
            algorithm.Lambda = lambda;
            sprsAprx = IterativeSparseApproximater(...
                'Algorithm', algorithm,...
                'MaxIter',4);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateGaFmin();
            
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
            options = optimoptions(options,'MaxIter',4);
            [~, costPst] = testCase.target.step(srcImgs,options);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre*1.01));
            
        end
        
        function testStepIstaVolumetricDataChannel(testCase,...
                sch,ach)
            
            % Configuration
            nDecs = [2 2 2];
            nChs  = [sch ach];
            nOrd  = [2 2 2];
            nVms  = 1;
            
            % Parameters
            lambda = 1e-3;
            srcImgs{1} = randn(16,16,16)+0.5;
            
            % Expected value
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            algorithm = FistaSystem();
            algorithm.Lambda = lambda;
            sprsAprx = IterativeSparseApproximater(...
                'Algorithm', algorithm,...
                'MaxIter',4);
            import saivdr.dictionary.nsoltx.design.*
            dicUpd =  NsoltDictionaryUpdateGaFmin();
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.*
            testCase.target = NsoltDictionaryLearningPnP(...
                'DataType','Volumetric Data',...
                'SparseApproximater',sprsAprx,...
                'DictionaryUpdater', dicUpd,...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'NumberOfVanishingMoments',nVms,...
                'PolyPhaseOrder',nOrd);
            
            % Preparation
            lppufbPre = testCase.target.OvsdLpPuFb;
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(lppufbPre);
            analyzer    = NsoltFactory.createAnalysis3dSystem(lppufbPre);
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
            options = optimoptions(options,'MaxIter',4);
            [~, costPst] = testCase.target.step(srcImgs,options);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            testCase.verifyThat(costPst, IsLessThanOrEqualTo(costPre*1.01));
            
        end
        
        
    end
    
end
