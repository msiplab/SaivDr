classdef MixtureOfUnitarySynthesisSystemTestCase < matlab.unittest.TestCase
    %MIXTUREOFUNITARYSYNTHESISSYSTEMTESTCASE Test case for MixtureOFUNITARYSynthesisSystem
    %
    % SVN identifier:
    % $Id: MixtureOfUnitarySynthesisSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        synthesizer
    end
    
    methods (TestMethodTeardown)
        function deteleObject(testCase)
            delete(testCase.synthesizer);
        end
    end
    
    methods (Test)
        
        % Test
        function testTwoMixtureCase(testCase)
            
            % Parameter setting
            subScales = [
                2 2 ;
                2 2 ;
                2 2 ;
                2 2 ;
                4 4 ;
                4 4 ;
                4 4 ;
                8 8 ;
                8 8 ;
                8 8 ];
            nCoefs = subScales(:,1).' * subScales(:,2);
            subCoefs{1} = rand(1,nCoefs);
            subCoefs{2} = rand(1,nCoefs);
            
            % Preparation
            import saivdr.dictionary.nsgenlotx.*
            import saivdr.dictionary.nsoltx.*
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'OutputMode','ParameterMatrixSet');
            subSynthesizers{1} = NsoltFactory.createSynthesis2dSystem(fb_);
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'TvmAngleInDegree',30,...
                'OutputMode','ParameterMatrixSet');
            subSynthesizers{2} = NsoltFactory.createSynthesis2dSystem(fb_);
            
            normFactor = 1/sqrt(length(subSynthesizers));
            
            % Expected value
            subRecImg{1} = step(subSynthesizers{1},...
                subCoefs{1}, subScales);
            subRecImg{2} = step(subSynthesizers{2},...
                subCoefs{2}, subScales);
            recImgExpctd  = normFactor * (subRecImg{1} + subRecImg{2});
            
            % Instantiation
            import saivdr.dictionary.mixture.*
            testCase.synthesizer = MixtureOfUnitarySynthesisSystem(...
                'UnitarySynthesizerSet',subSynthesizers);
            
            % Actual value
            coefs  = cell2mat(subCoefs);
            scalesInCell = cell(2,1);
            scalesInCell{1} = [ subScales ; -1 -1 ];
            scalesInCell{2} = [ subScales ];
            scales = cell2mat(scalesInCell);
            recImgActual = step(testCase.synthesizer,coefs, scales);
            
            % Evaluation
            diff = max(abs(recImgExpctd(:)-recImgActual(:))...
                ./abs(recImgExpctd(:)));
            testCase.verifyEqual(recImgActual,recImgExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
        end
        
        % Test
        function testThreeMixtureCase(testCase)
            
            % Parameter setting
            scalesInCell = cell(3,1);
            scalesInCell{1} = [
                8 8;
                8 8;
                8 8;
                8 8];
            scalesInCell{2} = [
                4 4;
                4 4;
                4 4;
                4 4;
                8 8;
                8 8;
                8 8];
            scalesInCell{3} = [
                2 2;
                2 2;
                2 2;
                2 2;
                4 4;
                4 4;
                4 4;
                8 8;
                8 8;
                8 8];
            subCoefs{1} = rand(1,16*16);
            subCoefs{2} = rand(1,16*16);
            subCoefs{3} = rand(1,16*16);
            
            % Preparation
            import saivdr.dictionary.nsgenlotx.*
            import saivdr.dictionary.nsoltx.*
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'OutputMode','ParameterMatrixSet');
            subSynthesizers{1} = NsoltFactory.createSynthesis2dSystem(fb_);
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'TvmAngleInDegree',30,...
                'OutputMode','ParameterMatrixSet');
            subSynthesizers{2} = NsoltFactory.createSynthesis2dSystem(fb_);
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'TvmAngleInDegree',60,...
                'OutputMode','ParameterMatrixSet');
            subSynthesizers{3} = NsoltFactory.createSynthesis2dSystem(fb_);
            
            normFactor = 1/sqrt(length(subSynthesizers));
            
            % Expected value
            subRecImg{1} = step(subSynthesizers{1},...
                subCoefs{1}, scalesInCell{1});
            subRecImg{2} = step(subSynthesizers{2},...
                subCoefs{2}, scalesInCell{2});
            subRecImg{3} = step(subSynthesizers{3},...
                subCoefs{3}, scalesInCell{3});
            recImgExpctd  = normFactor * ...
                (subRecImg{1} + subRecImg{2} + subRecImg{3});
            
            % Instantiation
            import saivdr.dictionary.mixture.*
            testCase.synthesizer = MixtureOfUnitarySynthesisSystem(...
                'UnitarySynthesizerSet',subSynthesizers);
            
            % Actual value
            coefs  = cell2mat(subCoefs);
            scalesInCell{1} = [ scalesInCell{1} ; -1 -1 ];
            scalesInCell{2} = [ scalesInCell{2} ; -1 -1 ];
            scalesInCell{3} = [ scalesInCell{3} ];
            scales = cell2mat(scalesInCell);
            recImgActual = step(testCase.synthesizer,coefs, scales);
            
            % Evaluation
            diff = max(abs(recImgExpctd(:)-recImgActual(:))...
                ./abs(recImgExpctd(:)));
            testCase.verifyEqual(recImgActual,recImgExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
        end
        
        % Test
        function testFiveMixtureCase(testCase)
            
            % Parameter setting
            nDics = 5;
            subScales = [
                2 2 ;
                2 2 ;
                2 2 ;
                2 2 ;
                4 4 ;
                4 4 ;
                4 4 ;
                8 8 ;
                8 8 ;
                8 8 ];
            nCoefs = subScales(:,1).' * subScales(:,2);
            subCoefs = cell(1,nDics);
            for iDic = 1:nDics
                subCoefs{iDic} = rand(1,nCoefs);
            end
            
            % Preparation
            import saivdr.dictionary.nsgenlotx.*
            import saivdr.dictionary.nsoltx.*
            subSynthesizers = cell(1,nDics);
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'OutputMode','ParameterMatrixSet');
            subSynthesizers{1} = NsoltFactory.createSynthesis2dSystem(fb_);
            %
            phi = [ -30 30 60 120 ];
            for iDic = 2:nDics
                fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                    'NumberOfVanishingMoments',2,...
                    'TvmAngleInDegree',phi(iDic-1),...
                    'OutputMode','ParameterMatrixSet');
                subSynthesizers{iDic} = ...
                    NsoltFactory.createSynthesis2dSystem(fb_);
            end
            
            normFactor = 1/sqrt(length(subSynthesizers));
            
            % Expected value
            recImgExpctd = 0;
            for iDic = 1:nDics
                recImgExpctd = recImgExpctd + ....
                    step(subSynthesizers{iDic},subCoefs{iDic}, subScales);
            end
            recImgExpctd  = normFactor * recImgExpctd;
            
            % Instantiation
            import saivdr.dictionary.mixture.*
            testCase.synthesizer = MixtureOfUnitarySynthesisSystem(...
                'UnitarySynthesizerSet',subSynthesizers);
            
            % Actual value
            coefs  = cell2mat(subCoefs);
            scalesInCell = cell(nDics,1);
            for iDic = 1:nDics-1
                scalesInCell{iDic} = [ subScales ; -1 -1 ];
            end
            iDic = nDics;
            scalesInCell{iDic} = subScales;
            scales = cell2mat(scalesInCell);
            recImgActual = step(testCase.synthesizer,coefs, scales);
            
            % Evaluation
            diff = max(abs(recImgExpctd(:)-recImgActual(:))...
                ./abs(recImgExpctd(:)));
            testCase.verifyEqual(recImgActual,recImgExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
        end
        
        % Test
        function testClone(testCase)
            
            % Parameter setting
            nDics = 5;
            subScales = [
                2 2 ;
                2 2 ;
                2 2 ;
                2 2 ;
                4 4 ;
                4 4 ;
                4 4 ;
                8 8 ;
                8 8 ;
                8 8 ];
            nCoefs = subScales(:,1).' * subScales(:,2);
            subCoefs = cell(1,nDics);
            for iDic = 1:nDics
                subCoefs{iDic} = rand(1,nCoefs);
            end
            coefs  = cell2mat(subCoefs);
            scalesInCell = cell(nDics,1);
            for iDic = 1:nDics-1
                scalesInCell{iDic} = [ subScales ; -1 -1 ];
            end
            iDic = nDics;
            scalesInCell{iDic} = subScales;
            scales = cell2mat(scalesInCell);            
            
            % Preparation
            import saivdr.dictionary.nsgenlotx.*
            import saivdr.dictionary.nsoltx.*
            subSynthesizers = cell(1,nDics);
            %
            fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',2,...
                'OutputMode','ParameterMatrixSet');
            subSynthesizers{1} = NsoltFactory.createSynthesis2dSystem(fb_);
            %
            phi = [ -30 30 60 120 ];
            for iDic = 2:nDics
                fb_ = NsGenLotFactory.createLpPuFb2dSystem(...
                    'NumberOfVanishingMoments',2,...
                    'TvmAngleInDegree',phi(iDic-1),...
                    'OutputMode','ParameterMatrixSet');
                subSynthesizers{iDic} = ...
                    NsoltFactory.createSynthesis2dSystem(fb_);
            end
            
            % Instantiation
            import saivdr.dictionary.mixture.*
            testCase.synthesizer = MixtureOfUnitarySynthesisSystem(...
                'UnitarySynthesizerSet',subSynthesizers);
            
            % Clone
            cloneSynthesizer = clone(testCase.synthesizer);
            
            % Evaluation
            testCase.verifyEqual(cloneSynthesizer,testCase.synthesizer);
            testCase.verifyFalse(cloneSynthesizer == testCase.synthesizer);
            prpOrg = get(testCase.synthesizer,'UnitarySynthesizerSet');
            prpCln = get(cloneSynthesizer,'UnitarySynthesizerSet');
            testCase.verifyEqual(prpCln,prpOrg);
            for iDic = 1:nDics
                testCase.verifyFalse(prpCln{iDic} == prpOrg{iDic});
            end
            %
            recImgExpctd = step(testCase.synthesizer,coefs,scales);
            recImgActual = step(cloneSynthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(recImgExpctd(:)-recImgActual(:))...
                ./abs(recImgExpctd(:)));
            testCase.verifyEqual(recImgActual,recImgExpctd,'RelTol',1e-15,...
                sprintf('%g',diff));
            
        end
    end
end
