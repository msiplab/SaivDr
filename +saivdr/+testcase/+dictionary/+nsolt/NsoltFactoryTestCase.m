classdef NsoltFactoryTestCase < matlab.unittest.TestCase
    %NSOLTFACTORYTESTCASE Test case for NsoltFactory
    %
    % SVN identifier:
    % $Id: NsoltFactoryTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        obj;
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.obj);
        end
    end
    
    methods (Test)
        
        function testCreateOvsdLpPuFb2dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dDec33(testCase)
            % Parameters
            dec = [3 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dDec22(testCase)
            % Parameters
            dec = [2 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dDec11Ch4(testCase)
            % Parameters
            decch = [1 1 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dDec11Ch5(testCase)
            % Parameters
            decch = [1 1 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dDec22Ch4(testCase)
            % Parameters
            decch = [2 2 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dDec22Ch5(testCase)
            % Parameters
            decch = [2 2 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dDeepCopy(testCase)
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem();
            cloneObj = NsoltFactory.createOvsdLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneObj,'Angles')));
            set(cloneObj,'Angles',angles);
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            import matlab.unittest.constraints.IsGreaterThan;
            testCase.verifyThat(norm(coefExpctd(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
        end
        
        function testCreateOvsdLpPuFb2dDeepCopyDec33(testCase)
            
            % Parameters
            decch = [3 3];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            cloneObj = NsoltFactory.createOvsdLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneObj,'Angles')));
            set(cloneObj,'Angles',angles);
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            import matlab.unittest.constraints.IsGreaterThan
            testCase.verifyThat(norm(coefExpctd(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
        end
        
        function testCreateOvsdLpPuFb2dDec22Ch4Vm0(testCase)
            
            % Parameters
            decch = [2 2 4];
            vm = 0;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIVm0System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'NumberOfVanishingMoments',vm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dDec22Ch5Vm0(testCase)
            
            % Parameters
            decch = [2 2 5];
            vm = 0;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIIVm0System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'NumberOfVanishingMoments',vm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dTypeIVm0DeepCopy(testCase)
            
            vm = 0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[2 2],...
                'NumberOfVanishingMoments',vm);
            cloneObj = NsoltFactory.createOvsdLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIVm0System';
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneObj,'Angles')));
            set(cloneObj,'Angles',angles);
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            import matlab.unittest.constraints.IsGreaterThan;
            testCase.verifyThat(norm(coefExpctd(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dTypeIIVm0DeepCopy(testCase)
            
            vm = 0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[2 2],...
                'NumberOfChannels',5,...
                'NumberOfVanishingMoments',vm);
            cloneObj = NsoltFactory.createOvsdLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIIVm0System';
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneObj,'Angles')));
            set(cloneObj,'Angles',angles);
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            import matlab.unittest.constraints.IsGreaterThan
            testCase.verifyThat(norm(coefExpctd(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dDec22Ch4Vm1(testCase)
            
            % Parameters
            decch = [2 2 4];
            vm = 1;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'NumberOfVanishingMoments',vm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dDec22Ch5Vm1(testCase)
            
            % Parameters
            decch = [2 2 5];
            vm = 1;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'NumberOfVanishingMoments',vm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dTypeIVm1DeepCopy(testCase)
            
            vm = 1;
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[ 2 2 ],...
                'NumberOfVanishingMoments',vm);
            cloneObj = NsoltFactory.createOvsdLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIVm1System';
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneObj,'Angles')));
            set(cloneObj,'Angles',angles);
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            import matlab.unittest.constraints.IsGreaterThan;
            testCase.verifyThat(norm(coefExpctd(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb2dTypeIIVm1DeepCopy(testCase)
            
            vm = 1;
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[2 2],...
                'NumberOfChannels',5,...
                'NumberOfVanishingMoments',vm);
            cloneObj = NsoltFactory.createOvsdLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            classExpctd = 'saivdr.dictionary.nsolt.OvsdLpPuFb2dTypeIIVm1System';
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneObj,'Angles')));
            set(cloneObj,'Angles',angles);
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            import matlab.unittest.constraints.IsGreaterThan;
            testCase.verifyThat(norm(coefExpctd(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysisSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIAnalysisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createAnalysisSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysisSystemDec33(testCase)
            % Parameters
            decch = [3 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIIAnalysisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = NsoltFactory.createAnalysisSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysisSystemDec22(testCase)
            % Parameters
            decch = [2 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIAnalysisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = NsoltFactory.createAnalysisSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysisSystemDec11Ch4(testCase)
            % Parameters
            decch = [1 1 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIAnalysisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = NsoltFactory.createAnalysisSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysisSystemDec11Ch5(testCase)
            % Parameters
            decch = [1 1 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIIAnalysisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = NsoltFactory.createAnalysisSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysisSystemDec22Ch4(testCase)
            % Parameters
            decch = [2 2 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIAnalysisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = NsoltFactory.createAnalysisSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysisSystemDec22Ch5(testCase)
            % Parameters
            decch = [2 2 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIIAnalysisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = NsoltFactory.createAnalysisSystem(lppufb);

            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        %{
         function testInvalidArguments(testCase)
            
            % Parameters
            
            % Expected values
            exceptionIdExpctd = 'SymTif:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Unsupported ...');
            
            % Instantiation of target class
            try
                nsolt.NsoltFactory.createOvsdLpPuFb2dSystem(...);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdExpctd, exceptionIdActual);
                messageActual = me.message;
                testCase.verifyEqual(messageExpctd, messageActual);
            end
        end
       %}
        
        function testCreateAnalysisSystemLpPuFb2d(testCase)
            
            %
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem();
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIAnalysisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createAnalysisSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysisSystemLpPuFb2dDec33(testCase)
            
            %
            decch = [ 3 3 ];
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIIAnalysisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createAnalysisSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesisSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeISynthesisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.obj = NsoltFactory.createSynthesisSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesisSystemDec33(testCase)
            % Parameters
            decch = [3 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIISynthesisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = NsoltFactory.createSynthesisSystem(lppufb);

            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesisSystemDec22(testCase)
            % Parameters
            decch = [2 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeISynthesisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = NsoltFactory.createSynthesisSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesisSystemDec11Ch4(testCase)
            % Parameters
            decch = [1 1 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeISynthesisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = NsoltFactory.createSynthesisSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesisSystemDec11Ch5(testCase)
            % Parameters
            decch = [1 1 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIISynthesisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = NsoltFactory.createSynthesisSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesisSystemDec22Ch4(testCase)
            % Parameters
            decch = [2 2 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeISynthesisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = NsoltFactory.createSynthesisSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesisSystemDec22Ch5(testCase)
            % Parameters
            decch = [2 2 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIISynthesisSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = NsoltFactory.createSynthesisSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        
        
        function testCreateSynthesisSystemLpPuFb2d(testCase)
            
            %
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem();
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeISynthesisSystem';
            
            % Instantiation of target class
            testCase.obj = NsoltFactory.createSynthesisSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesisSystemLpPuFb2dDec33(testCase)
            
            %
            decch = [ 3 3 ];
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            
            % Expected values
            classExpctd = 'saivdr.dictionary.nsolt.TypeIISynthesisSystem';
            
            % Instantiation of target class
            testCase.obj = NsoltFactory.createSynthesisSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesisSystemLpPuFb2dParameterMatrixSet(testCase)
            
            %
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'OutputMode','ParameterMatrixSet');
            % Expected values
            outputExpctd = 'ParameterMatrixSet';
            
            % Instantiation of target class
            testCase.obj = NsoltFactory.createSynthesisSystem(lppufb);
            
            % Actual values
            outputActual = get(lppufb,'OutputMode');
            
            % Evaluation
            testCase.verifyEqual(outputActual,outputExpctd);
            
        end        
        
        function testCreateSynthesisSystemLpPuFb2dWithoutOutputMode(testCase)
            
            %
            import saivdr.dictionary.nsolt.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem();
            % Expected values
            outputExpctd = 'Coefficients';
            
            % Instantiation of target class
            testCase.obj = NsoltFactory.createSynthesisSystem(lppufb);
            
            % Actual values
            outputActual = get(lppufb,'OutputMode');
            
            % Evaluation
            testCase.verifyEqual(outputActual,outputExpctd);
            
        end          
        
        function testCreateSynthesisSystemIsCloneLpPuFb2dFalse(testCase)
            
            %
            import saivdr.dictionary.nsolt.*
            
            % Instantiation of target class
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem();
            testCase.obj = NsoltFactory.createSynthesisSystem(lppufb,...
                'IsCloneLpPuFb2d',false);
            
            % Actual values
            flagActual = get(testCase.obj,'IsCloneLpPuFb2d');
            
            % Evaluation
            testCase.verifyFalse(flagActual);
            
        end            
        
        function testCreateAnalysisSystemIsCloneLpPuFb2dFalse(testCase)
            
            %
            import saivdr.dictionary.nsolt.*
            
            % Instantiation of target class
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem();
            testCase.obj = NsoltFactory.createAnalysisSystem(lppufb,...
                'IsCloneLpPuFb2d',false);
            
            % Actual values
            flagActual = get(testCase.obj,'IsCloneLpPuFb2d');
            
            % Evaluation
            testCase.verifyFalse(flagActual);
            
        end          
        
        %{
         function testInvalidArguments(testCase)
            
            % Parameters
            
            % Expected values
            exceptionIdExpctd = 'SymTif:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Unsupported ...');
            
            % Instantiation of target class
            try
                nsolt.NsoltFactory.createOvsdLpPuFb2dSystem(...);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdExpctd, exceptionIdActual);
                messageActual = me.message;
                testCase.verifyEqual(messageExpctd, messageActual);
            end
        end
        %}
    end
end


