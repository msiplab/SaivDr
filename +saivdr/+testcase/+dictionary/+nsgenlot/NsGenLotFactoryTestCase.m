classdef NsGenLotFactoryTestCase < matlab.unittest.TestCase
    %NSGENLOTFACTORYTESTCASE Test case for NsGenLotFactory
    %
    % SVN identifier:
    % $Id: NsGenLotFactoryTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        obj
    end
    
    methods (TestMethodTeardown)
        
        function tearDown(testCase)
            delete(testCase.obj)
        end
    end
    
    methods (Test)
        
        function testCreateLpPuFb2dVm0(testCase)
            
            % Parameters
            nVm = 0;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIVm0System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*
            testCase.obj = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',nVm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            
        end
        
        function testCreateLpPuFb2dVm1(testCase)
            
            % Parameters
            nVm = 1;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*
            testCase.obj = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',nVm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            
        end
        
        function testCreateLpPuFb2dVm2(testCase)
            
            % Parameters
            nVm = 2;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsgenlot.LpPuFb2dVm2System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*        
            testCase.obj = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',nVm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            
        end

        function testCreateLpPuFb2dTvm(testCase)
            
            % Parameters
            nVm = 2;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsgenlot.LpPuFb2dTvmSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*        
            testCase.obj = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',nVm,...
                'TvmAngleInDegree',0);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            
        end

        function testInvalidArguments(testCase)
            
            % Parameters
            nVm = -1;
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Unsupported type of vanishing moments');
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*        
            try
                testCase.obj = NsGenLotFactory.createLpPuFb2dSystem(...
                    'NumberOfVanishingMoments',nVm);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
        end
        
        function testCreateLpPuFb2dVm0DeepCopy(testCase)
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*        
            testCase.obj = NsGenLotFactory.createLpPuFb2dSystem();
            cloneLpPuFb = NsGenLotFactory.createLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            
            % Actual values
            coefActual = step(cloneLpPuFb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,...
                sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneLpPuFb,'Angles')));
            
            % Actual values
            coefActual = step(cloneLpPuFb,angles,[]);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThan(1e-15),...
                sprintf('%g',coefDist));
            
        end

        function testCreateLpPuFb2dDec44Ord22Vm2DeepCopy(testCase)
            
            % Parameters
            nDecs = [ 4 4 ];
            nOrds = [ 2 2 ];
            ang = 2*pi*rand(28,6);
            mus = 2*round(rand(8,6))-1;
            nVm = 2;
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*        
            testCase.obj = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',nVm,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            cloneLpPuFb = NsGenLotFactory.createLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,ang,mus);
            
            % Actual values
            coefActual = step(cloneLpPuFb,ang,mus);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./...
                abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,...
                sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneLpPuFb,'Angles')));
            
            % Actual values
            coefActual = step(cloneLpPuFb,angles,[]);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./...
                abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThan(1e-15),...
                sprintf('%g',coefDist));
            
        end
                                
        function testCreateLpPuFb2dDec44Ord22Vm1DeepCopy(testCase)
            
            % Parameters
            nDecs = [ 4 4 ];
            nOrds = [ 2 2 ];
            ang = 2*pi*rand(28,6);
            mus = 2*round(rand(8,6))-1;
            nVm = 1;
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*        
            testCase.obj = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',nVm,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            cloneLpPuFb = NsGenLotFactory.createLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,ang,mus);
            
            % Actual values
            coefActual = step(cloneLpPuFb,ang,mus);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./...
                abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,...
                sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneLpPuFb,'Angles')));
                        
            % Actual values
            coefActual = step(cloneLpPuFb,angles,[]);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./...
                abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThan(1e-15),...
                sprintf('%g',coefDist));
            
        end
        
        function testCreateLpPuFb2dDec44Ord22Tvm120DeepCopy(testCase)
            
            % Parameters
            nDecs = [ 4 4 ];
            nOrds = [ 2 2 ];
            ang = 2*pi*rand(28,6);
            mus = 2*round(rand(8,6))-1;
            nVm = 2;
            aTvm = 120;
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*        
            testCase.obj = NsGenLotFactory.createLpPuFb2dSystem(...
                'NumberOfVanishingMoments',nVm,...
                'TvmAngleInDegree',aTvm,...
                'DecimationFactor',nDecs,...
                'PolyPhaseOrder',nOrds);
            cloneLpPuFb = NsGenLotFactory.createLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,ang,mus);
            
            % Actual values
            coefActual = step(cloneLpPuFb,ang,mus);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./...
                abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,...
                sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneLpPuFb,'Angles')));
            
            % Actual values
            coefActual = step(cloneLpPuFb,angles,[]);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./...
                abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThan(1e-15),...
                sprintf('%g',coefDist));
            
        end

    end
end
