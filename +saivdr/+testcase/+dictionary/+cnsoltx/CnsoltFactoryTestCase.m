classdef CnsoltFactoryTestCase < matlab.unittest.TestCase
    %NSOLTFACTORYTESTCASE Test case for CnsoltFactory
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2014-2016, Shogo MURAMATSU
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
        obj;
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.obj);
        end
    end
    
    methods (Test)
        
        function testCreateCplxOvsdLpPuFb2dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dDec33(testCase)
            % Parameters
            dec = [3 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dDec22(testCase)
            % Parameters
            dec = [2 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dDec11Ch4(testCase)
            % Parameters
            decch = [1 1 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dDec11Ch5(testCase)
            % Parameters
            decch = [1 1 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dDec22Ch4(testCase)
            % Parameters
            decch = [2 2 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dDec22Ch5(testCase)
            % Parameters
            decch = [2 2 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dDeepCopy(testCase)
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem();
            cloneObj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(testCase.obj);
            
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
            
            % Original values
            coefOriginal= step(testCase.obj,[],[]);
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefOriginal(:)-coefActual(:))./abs(coefOriginal(:)));
            import matlab.unittest.constraints.IsGreaterThan;
            testCase.verifyThat(norm(coefOriginal(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
        end
        
        function testCreateCplxOvsdLpPuFb2dDeepCopyDec33(testCase)
            
            % Parameters
            decch = [3 3];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            cloneObj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(testCase.obj);
            
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

            % Original values
            coefOriginal= step(testCase.obj,[],[]);            
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefOriginal(:)-coefActual(:))./abs(coefOriginal(:)));
            import matlab.unittest.constraints.IsGreaterThan
            testCase.verifyThat(norm(coefOriginal(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
        end
        
        function testCreateCplxOvsdLpPuFb2dDec22Ch4Vm0(testCase)
            
            % Parameters
            decch = [2 2 4];
            vm = 0;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIVm0System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'NumberOfVanishingMoments',vm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dDec22Ch5Vm0(testCase)
            
            % Parameters
            decch = [2 2 5];
            vm = 0;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIIVm0System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'NumberOfVanishingMoments',vm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dTypeIVm0DeepCopy(testCase)
            
            vm = 0;
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',[2 2],...
                'NumberOfVanishingMoments',vm);
            cloneObj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIVm0System';
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneObj,'Angles')));
            set(cloneObj,'Angles',angles);
            
            % Original values
            coefOriginal= step(testCase.obj,[],[]);            
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefOriginal(:)-coefActual(:))./abs(coefOriginal(:)));
            import matlab.unittest.constraints.IsGreaterThan;
            testCase.verifyThat(norm(coefOriginal(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dTypeIIVm0DeepCopy(testCase)
            
            vm = 0;
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',[2 2],...
                'NumberOfChannels',5,...
                'NumberOfVanishingMoments',vm);
            cloneObj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIIVm0System';
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneObj,'Angles')));
            set(cloneObj,'Angles',angles);
            
            % Original values
            coefOriginal = step(testCase.obj,[],[]);            
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefOriginal(:)-coefActual(:))./abs(coefOriginal(:)));
            import matlab.unittest.constraints.IsGreaterThan
            testCase.verifyThat(norm(coefOriginal(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dDec22Ch4Vm1(testCase)
            
            % Parameters
            decch = [2 2 4];
            vm = 1;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'NumberOfVanishingMoments',vm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dDec22Ch5Vm1(testCase)
            
            % Parameters
            decch = [2 2 5];
            vm = 1;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'NumberOfVanishingMoments',vm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dTypeIVm1DeepCopy(testCase)
            
            vm = 1;
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',[ 2 2 ],...
                'NumberOfVanishingMoments',vm);
            cloneObj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIVm1System';
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneObj,'Angles')));
            set(cloneObj,'Angles',angles);
            
            % Original values
            coefOriginal = step(testCase.obj,[],[]);            
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefOriginal(:)-coefActual(:))./abs(coefOriginal(:)));
            import matlab.unittest.constraints.IsGreaterThan;
            testCase.verifyThat(norm(coefOriginal(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateCplxOvsdLpPuFb2dTypeIIVm1DeepCopy(testCase)
            
            vm = 1;
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',[2 2],...
                'NumberOfChannels',5,...
                'NumberOfVanishingMoments',vm);
            cloneObj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIIVm1System';
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneObj,'Angles')));
            set(cloneObj,'Angles',angles);
            
            % Original values
            coefOriginal = step(testCase.obj,[],[]);            
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefOriginal(:)-coefActual(:))./abs(coefOriginal(:)));
            import matlab.unittest.constraints.IsGreaterThan;
            testCase.verifyThat(norm(coefOriginal(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis2dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createAnalysis2dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis2dSystemDec33(testCase)
            % Parameters
            decch = [3 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = CnsoltFactory.createAnalysis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis2dSystemDec22(testCase)
            % Parameters
            decch = [2 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = CnsoltFactory.createAnalysis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis2dSystemDec11Ch4(testCase)
            % Parameters
            decch = [1 1 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = CnsoltFactory.createAnalysis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis2dSystemDec11Ch5(testCase)
            % Parameters
            decch = [1 1 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = CnsoltFactory.createAnalysis2dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis2dSystemDec22Ch4(testCase)
            % Parameters
            decch = [2 2 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = CnsoltFactory.createAnalysis2dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis2dSystemDec22Ch5(testCase)
            % Parameters
            decch = [2 2 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = CnsoltFactory.createAnalysis2dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis2dSystemLpPuFb2d(testCase)
            
            %
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem();
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createAnalysis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis2dSystemLpPuFb2dDec33(testCase)
            
            %
            decch = [ 3 3 ];
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createAnalysis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis2dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createSynthesis2dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis2dSystemDec33(testCase)
            % Parameters
            decch = [3 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis2dSystemDec22(testCase)
            % Parameters
            decch = [2 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis2dSystemDec11Ch4(testCase)
            % Parameters
            decch = [1 1 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis2dSystemDec11Ch5(testCase)
            % Parameters
            decch = [1 1 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis2dSystemDec22Ch4(testCase)
            % Parameters
            decch = [2 2 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis2dSystemDec22Ch5(testCase)
            % Parameters
            decch = [2 2 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis2dSystemLpPuFb2d(testCase)
            
            %
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem();
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis2dSystem';
            
            % Instantiation of target class
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis2dSystemLpPuFb2dDec33(testCase)
            
            %
            decch = [ 3 3 ];
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis2dSystem';
            
            % Instantiation of target class
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis2dSystemLpPuFb2dParameterMatrixSet(testCase)
            
            %
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'OutputMode','ParameterMatrixSet');
            % Expected values
            outputExpctd = 'ParameterMatrixSet';
            
            % Instantiation of target class
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb);
            
            % Actual values
            outputActual = get(lppufb,'OutputMode');
            
            % Evaluation
            testCase.verifyEqual(outputActual,outputExpctd);
            
        end
        
        function testCreateSynthesis2dSystemLpPuFb2dWithoutOutputMode(testCase)
            
            %
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem();
            % Expected values
            outputExpctd = 'Coefficients';
            
            % Instantiation of target class
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb);
            
            % Actual values
            outputActual = get(lppufb,'OutputMode');
            
            % Evaluation
            testCase.verifyEqual(outputActual,outputExpctd);
            
        end
        
        function testCreateSynthesis2dSystemIsCloneLpPuFb2dFalse(testCase)
            
            %
            import saivdr.dictionary.cnsoltx.*
            
            % Instantiation of target class
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem();
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb,...
                'IsCloneLpPuFb2d',false);
            
            % Actual values
            flagActual = get(testCase.obj,'IsCloneLpPuFb2d');
            
            % Evaluation
            testCase.verifyFalse(flagActual);
            
        end
        
        function testCreateAnalysis2dSystemIsCloneLpPuFb2dFalse(testCase)
            
            %
            import saivdr.dictionary.cnsoltx.*
            
            % Instantiation of target class
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem();
            testCase.obj = CnsoltFactory.createAnalysis2dSystem(lppufb,...
                'IsCloneLpPuFb2d',false);
            
            % Actual values
            flagActual = get(testCase.obj,'IsCloneLpPuFb2d');
            
            % Evaluation
            testCase.verifyFalse(flagActual);
            
        end
        
        function testCreateCplxOvsdLpPuFb3dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb3dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb3dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end        
        
       
        function testCreateAnalysis3dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis3dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createAnalysis3dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis3dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis3dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createSynthesis3dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
                
      
        function testCreateCplxOvsdLpPuFb2dDec22Ch32(testCase)
            % Parameters
            dec = [2 2];
            chs = [3 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
      
        function testCreateCplxOvsdLpPuFb2dDec22Ch23(testCase)
            % Parameters
            dec = [2 2];
            chs = [2 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb2dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end        
        
      
        function testCreateCplxOvsdLpPuFb3dDec222Ch54(testCase)
            % Parameters
            dec = [2 2 2];
            chs = [5 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb3dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end                
                 
      
        function testCreateCplxOvsdLpPuFb3dDec222Ch45(testCase)
            % Parameters
            dec = [2 2 2];
            chs = [4 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CplxOvsdLpPuFb3dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end                
        
       
        function testCreateAnalysis2dSystemDec22Ch32(testCase)
            % Parameters
            dec = [2 2];
            chs = [3 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            testCase.obj = CnsoltFactory.createAnalysis2dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis2dSystemDec22Ch23(testCase)
            % Parameters
            dec = [2 2];
            chs = [2 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            testCase.obj = CnsoltFactory.createAnalysis2dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end        
        
        function testCreateAnalysis3dSystemDec222Ch54(testCase)
            % Parameters
            dec = [2 2 2];
            chs = [5 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis3dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            testCase.obj = CnsoltFactory.createAnalysis3dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end        
        
        function testCreateAnalysis3dSystemDec222Ch45(testCase)
            % Parameters
            dec = [2 2 2];
            chs = [4 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltAnalysis3dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            testCase.obj = CnsoltFactory.createAnalysis3dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end                
        
        function testCreateSynthesis2dSystemDec22Ch32(testCase)
            % Parameters
            dec = [2 2];
            chs = [3 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis2dSystemDec22Ch23(testCase)
            % Parameters
            dec = [2 2];
            chs = [2 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis2dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            testCase.obj = CnsoltFactory.createSynthesis2dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis3dSystemDec222Ch54(testCase)
            % Parameters
            dec = [2 2 2];
            chs = [5 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis3dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            testCase.obj = CnsoltFactory.createSynthesis3dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end                  
        
        function testCreateSynthesis3dSystemDec222Ch45(testCase)
            % Parameters
            dec = [2 2 2];
            chs = [4 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.cnsoltx.CnsoltSynthesis3dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            lppufb = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            testCase.obj = CnsoltFactory.createSynthesis3dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
   
        function testCreateCplxOvsdLpPuFb3dDeepCopy(testCase)
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.obj = CnsoltFactory.createCplxOvsdLpPuFb3dSystem();
            cloneObj = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(testCase.obj);
            
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
            
            % Original values
            coefOriginal= step(testCase.obj,[],[]);
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefOriginal(:)-coefActual(:))./abs(coefOriginal(:)));
            import matlab.unittest.constraints.IsGreaterThan;
            testCase.verifyThat(norm(coefOriginal(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
        end
        
    end
    
end
