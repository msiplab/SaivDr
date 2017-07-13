classdef OLpPrFbFactoryTestCase < matlab.unittest.TestCase
    %OLpPrFbFactoryTESTCASE Test case for OLpPrFbFactory
    %
    % Requirements: MATLAB R2015b
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
        
        function testCreateOvsdLpPuFb1dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb1dDec5(testCase)
            % Parameters
            dec = 5;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb1dDec4(testCase)
            % Parameters
            dec = 4;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb1dDec1Ch4(testCase)
            % Parameters
            decch = [1 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb1dDec1Ch5(testCase)
            % Parameters
            decch = [1 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb1dDec4Ch4(testCase)
            % Parameters
            decch = [4 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb1dDec4Ch5(testCase)
            % Parameters
            decch = [4 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb1dDeepCopy(testCase)
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem();
            cloneObj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(testCase.obj);
            
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
            coefOriginal = step(testCase.obj,[],[]);            
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefOriginal(:)-coefActual(:))./abs(coefOriginal(:)));
            import matlab.unittest.constraints.IsGreaterThan;
            testCase.verifyThat(norm(coefOriginal(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
        end
        
        function testCreateOvsdLpPuFb1dDeepCopyDec5(testCase)
            
            % Parameters
            decch = [5 5];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            cloneObj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(testCase.obj);
            
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
            coefOriginal = step(testCase.obj,[],[]);                        
            
            % Actual values
            coefActual = step(cloneObj,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefOriginal(:)-coefActual(:))./abs(coefOriginal(:)));
            import matlab.unittest.constraints.IsGreaterThan
            testCase.verifyThat(norm(coefOriginal(:)-coefActual(:)),IsGreaterThan(1e-15),sprintf('%g',coefDist));
            
        end
        
        function testCreateOvsdLpPuFb1dDec4Ch4Vm0(testCase)
            
            % Parameters
            decch = [4 4];
            vm = 0;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIVm0System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'NumberOfVanishingMoments',vm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb1dDec4Ch5Vm0(testCase)
            
            % Parameters
            decch = [4 5];
            vm = 0;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIIVm0System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'NumberOfVanishingMoments',vm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb1dTypeIVm0DeepCopy(testCase)
            
            vm = 0;
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',4,...
                'NumberOfVanishingMoments',vm);
            cloneObj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIVm0System';
            
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
        
        function testCreateOvsdLpPuFb1dTypeIIVm0DeepCopy(testCase)
            
            vm = 0;
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',4,...
                'NumberOfChannels',5,...
                'NumberOfVanishingMoments',vm);
            cloneObj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIIVm0System';
            
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
        
        function testCreateOvsdLpPuFb1dDec4Ch4Vm1(testCase)
            
            % Parameters
            decch = [4 4];
            vm = 1;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'NumberOfVanishingMoments',vm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb1dDec4Ch5Vm1(testCase)
            
            % Parameters
            decch = [4 5];
            vm = 1;
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'NumberOfVanishingMoments',vm);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateOvsdLpPuFb1dTypeIVm1DeepCopy(testCase)
            
            vm = 1;
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',4,...
                'NumberOfVanishingMoments',vm);
            cloneObj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIVm1System';
            
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
        
        function testCreateOvsdLpPuFb1dTypeIIVm1DeepCopy(testCase)
            
            vm = 1;
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',4,...
                'NumberOfChannels',5,...
                'NumberOfVanishingMoments',vm);
            cloneObj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(testCase.obj);
            
            % Expected values
            coefExpctd = step(testCase.obj,[],[]);
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIIVm1System';
            
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
        
        %{
        function testCreateAnalysis1dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltAnalysis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createAnalysis1dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis1dSystemDec33(testCase)
            % Parameters
            decch = [3 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltAnalysis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            testCase.obj = OLpPrFbFactory.createAnalysis1dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis1dSystemDec22(testCase)
            % Parameters
            decch = [2 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltAnalysis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            testCase.obj = OLpPrFbFactory.createAnalysis1dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis1dSystemDec11Ch4(testCase)
            % Parameters
            decch = [1 1 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltAnalysis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            testCase.obj = OLpPrFbFactory.createAnalysis1dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis1dSystemDec11Ch5(testCase)
            % Parameters
            decch = [1 1 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltAnalysis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            testCase.obj = OLpPrFbFactory.createAnalysis1dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis1dSystemDec22Ch4(testCase)
            % Parameters
            decch = [2 2 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltAnalysis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            testCase.obj = OLpPrFbFactory.createAnalysis1dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis1dSystemDec22Ch5(testCase)
            % Parameters
            decch = [2 2 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltAnalysis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            testCase.obj = OLpPrFbFactory.createAnalysis1dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis1dSystemLpPuFb1d(testCase)
            
            %
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem();
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltAnalysis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createAnalysis1dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis1dSystemLpPuFb1dDec33(testCase)
            
            %
            decch = [ 3 3 ];
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltAnalysis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createAnalysis1dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis1dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltSynthesis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createSynthesis1dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis1dSystemDec33(testCase)
            % Parameters
            decch = [3 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltSynthesis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            testCase.obj = OLpPrFbFactory.createSynthesis1dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis1dSystemDec22(testCase)
            % Parameters
            decch = [2 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltSynthesis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            testCase.obj = OLpPrFbFactory.createSynthesis1dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis1dSystemDec11Ch4(testCase)
            % Parameters
            decch = [1 1 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltSynthesis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            testCase.obj = OLpPrFbFactory.createSynthesis1dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis1dSystemDec11Ch5(testCase)
            % Parameters
            decch = [1 1 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltSynthesis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            testCase.obj = OLpPrFbFactory.createSynthesis1dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis1dSystemDec22Ch4(testCase)
            % Parameters
            decch = [2 2 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltSynthesis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            testCase.obj = OLpPrFbFactory.createSynthesis1dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis1dSystemDec22Ch5(testCase)
            % Parameters
            decch = [2 2 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltSynthesis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            testCase.obj = OLpPrFbFactory.createSynthesis1dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis1dSystemLpPuFb1d(testCase)
            
            %
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem();
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltSynthesis1dSystem';
            
            % Instantiation of target class
            testCase.obj = OLpPrFbFactory.createSynthesis1dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis1dSystemLpPuFb1dDec33(testCase)
            
            %
            decch = [ 3 3 ];
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end));
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltSynthesis1dSystem';
            
            % Instantiation of target class
            testCase.obj = OLpPrFbFactory.createSynthesis1dSystem(lppufb);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis1dSystemLpPuFb1dParameterMatrixSet(testCase)
            
            %
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'OutputMode','ParameterMatrixSet');
            % Expected values
            outputExpctd = 'ParameterMatrixSet';
            
            % Instantiation of target class
            testCase.obj = OLpPrFbFactory.createSynthesis1dSystem(lppufb);
            
            % Actual values
            outputActual = get(lppufb,'OutputMode');
            
            % Evaluation
            testCase.verifyEqual(outputActual,outputExpctd);
            
        end
        
        function testCreateSynthesis1dSystemLpPuFb1dWithoutOutputMode(testCase)
            
            %
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem();
            % Expected values
            outputExpctd = 'Coefficients';
            
            % Instantiation of target class
            testCase.obj = OLpPrFbFactory.createSynthesis1dSystem(lppufb);
            
            % Actual values
            outputActual = get(lppufb,'OutputMode');
            
            % Evaluation
            testCase.verifyEqual(outputActual,outputExpctd);
            
        end
        
        function testCreateSynthesis1dSystemIsCloneLpPuFb1dFalse(testCase)
            
            %
            import saivdr.dictionary.olpprfb.*
            
            % Instantiation of target class
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem();
            testCase.obj = OLpPrFbFactory.createSynthesis1dSystem(lppufb,...
                'IsCloneLpPuFb1d',false);
            
            % Actual values
            flagActual = get(testCase.obj,'IsCloneLpPuFb1d');
            
            % Evaluation
            testCase.verifyFalse(flagActual);
            
        end
        
        function testCreateAnalysis1dSystemIsCloneLpPuFb1dFalse(testCase)
            
            %
            import saivdr.dictionary.olpprfb.*
            
            % Instantiation of target class
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem();
            testCase.obj = OLpPrFbFactory.createAnalysis1dSystem(lppufb,...
                'IsCloneLpPuFb1d',false);
            
            % Actual values
            flagActual = get(testCase.obj,'IsCloneLpPuFb1d');
            
            % Evaluation
            testCase.verifyFalse(flagActual);
            
        end
        
        function testCreateOvsdLpPuFb3dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb3dTypeIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb3dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
       
        function testCreateAnalysis3dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltAnalysis3dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createAnalysis3dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateSynthesis3dSystem(testCase)
            
            % Parameters
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltSynthesis3dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createSynthesis3dSystem();
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
                
      
        function testCreateOvsdLpPuFb1dDec22Ch32(testCase)
            % Parameters
            dec = [2 2];
            chs = [3 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
      
        function testCreateOvsdLpPuFb1dDec22Ch23(testCase)
            % Parameters
            dec = [2 2];
            chs = [2 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb1dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
      
        function testCreateOvsdLpPuFb3dDec222Ch54(testCase)
            % Parameters
            dec = [2 2 2];
            chs = [5 4];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb3dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
                 
      
        function testCreateOvsdLpPuFb3dDec222Ch45(testCase)
            % Parameters
            dec = [2 2 2];
            chs = [4 5];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.OvsdLpPuFb3dTypeIIVm1System';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.obj = OLpPrFbFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
       
        function testCreateAnalysis1dSystemDec22Ch32(testCase)
            % Parameters
            dec = [2 2];
            chs = [3 2];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltAnalysis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            testCase.obj = OLpPrFbFactory.createAnalysis1dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end
        
        function testCreateAnalysis1dSystemDec22Ch23(testCase)
            % Parameters
            dec = [2 2];
            chs = [2 3];
            
            % Expected values
            classExpctd = 'saivdr.dictionary.olpprfb.NsoltAnalysis1dSystem';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            lppufb = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',chs);
            testCase.obj = OLpPrFbFactory.createAnalysis1dSystem(lppufb);
            
            
            % Actual values
            classActual = class(testCase.obj);
            
            % Evaluation
            testCase.verifyEqual(classActual,classExpctd);
            testCase.verifyClass(testCase.obj,classExpctd);
            
        end

        %}
    end
    
end
