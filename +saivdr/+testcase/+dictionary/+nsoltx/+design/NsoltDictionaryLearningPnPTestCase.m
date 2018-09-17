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
        
        function testSetPropertiesIhtVolumetricData(testCase)
            
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
            Gp
            Ista
        %}
    end
end
