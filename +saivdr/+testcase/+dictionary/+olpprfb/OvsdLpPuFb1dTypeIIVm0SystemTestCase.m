classdef OvsdLpPuFb1dTypeIIVm0SystemTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB1dTYPEIIVM0SYSTEMTESTCASE Test case for OvsdLpPuFb1dTypeIIVm0System
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015-2016, Shogo MURAMATSU
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
        lppufb;
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.lppufb);
        end
    end
    
    methods (Test)

        % Test for default construction
        function testConstructor(testCase)
            
            % Expected values
            coefExpctd = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0  0  0  0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System();
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for default construction
        function testConstructorWithDeepCopy(testCase)
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System();
            cloneLpPuFb = clone(testCase.lppufb);
            
            % Expected values
            coefExpctd = step(testCase.lppufb,[],[]);
            
            % Actual values
            coefActual = step(cloneLpPuFb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneLpPuFb,'Angles')));
            
            % Actual values
            coefActual = step(cloneLpPuFb,angles,[]);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec5Ord0(testCase)
            
            % Parameters
            dec = 5;
            ord = 0;
            
            % Expected values
            coefExpctd = [
                0.447213595499958   0.447213595499958   0.447213595499958   0.447213595499958   0.447213595499958
                0.511667273601693  -0.195439507584855  -0.632455532033676  -0.195439507584855   0.511667273601693
                0.195439507584855  -0.511667273601693   0.632455532033676  -0.511667273601693   0.195439507584855
                0.601500955007546   0.371748034460185   0.000000000000000  -0.371748034460184  -0.601500955007546
                0.371748034460185  -0.601500955007546  -0.000000000000000   0.601500955007546  -0.371748034460184
            ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec4Ch5Ord0(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord   = 0;
            
            % Expected values
            coefExpctd = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0  0  0  0 
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 2
        function testConstructorWithDec4Ch5Ord2Ang(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord   = 2;
            ang = 2*pi*rand(4,2);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction with order 4
        function testConstructorWithDec4Ch5Ord4Ang(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 4;
            ang = 2*pi*rand(4,3);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2
        function testConstructorWithDec5Ord2(testCase)
            
            % Parameters
            dec = 5;
            ord = 2;
            ang = 2*pi*rand(4,2);
            
            % Expected values
            nDec = dec;
            nChs = nDec;
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4
        function testConstructorWithDec5Ord4(testCase)
            
            % Parameters
            dec = 5;
            ord = 4;
            ang = 2*pi*rand(4,3);
            
            % Expected values
            nDec = dec;
            nChs = nDec;
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec4Ch7Ord0(testCase)
            
            % Parameters
            decch = [ 4 7 ];
            ord   = 0;
            
            % Expected values
            dimExpctd = [7 4];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            
        end
        
        % Test for construction
        function testConstructorWithDec4Ch9Ord0(testCase)
            
            % Parameters
            decch = [ 4 9 ];
            ord   = 0;
            
            % Expected values
            dimExpctd = [9 4];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec5Ch9Ord0Ang0(testCase)
            
            % Parameters
            decch = [ 5 9 ];
            ord   = 0;
            ang   = 0;
            
            % Expected values
            coefExpctd = [
                0.447213595499958   0.447213595499958   0.447213595499958   0.447213595499958   0.447213595499958
                0.511667273601693  -0.195439507584855  -0.632455532033676  -0.195439507584855   0.511667273601693
                0.195439507584855  -0.511667273601693   0.632455532033676  -0.511667273601693   0.195439507584855
                0 0 0 0 0 
                0 0 0 0 0 
                0.601500955007546   0.371748034460185   0.000000000000000  -0.371748034460184  -0.601500955007546
                0.371748034460185  -0.601500955007546  -0.000000000000000   0.601500955007546  -0.371748034460184
                0 0 0 0 0                 
                0 0 0 0 0                                 
                ];
                
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,...
                sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec9Ch9Ord0Ang(testCase)
            
            % Parameters
            decch = [ 9 9 ];
            ord = 0;
            angW = zeros(10,1);
            angU = 2*pi*rand(6,1);
            
            % Expected values
            import saivdr.dictionary.utility.*
            omgsW = OrthonormalMatrixGenerationSystem();
            omgsU = OrthonormalMatrixGenerationSystem();
            matrixW0 = step(omgsW,angW,1);
            matrixU0 = step(omgsU,angU,1);
            coefExpctd = ...
                blkdiag(matrixW0, matrixU0) * ...
                [
                0.333333333333333   0.333333333333333   0.333333333333333   0.333333333333333   0.333333333333333   0.333333333333333   0.333333333333333   0.333333333333333   0.333333333333333
                0.442975349592450   0.235702260395516  -0.081858535979315  -0.361116813613135  -0.471404520791032  -0.361116813613135  -0.081858535979315   0.235702260395516   0.442975349592450
                0.361116813613135  -0.235702260395516  -0.442975349592450   0.081858535979315   0.471404520791032   0.081858535979316  -0.442975349592450  -0.235702260395516   0.361116813613135
                0.235702260395516  -0.471404520791032   0.235702260395516   0.235702260395516  -0.471404520791032   0.235702260395515   0.235702260395516  -0.471404520791032   0.235702260395516
                0.081858535979315  -0.235702260395516   0.361116813613135  -0.442975349592450   0.471404520791032  -0.442975349592450   0.361116813613135  -0.235702260395515   0.081858535979316
                0.464242826880013   0.408248290463863   0.303012985114696   0.161229841765317   0.000000000000000  -0.161229841765317  -0.303012985114696  -0.408248290463863  -0.464242826880013
                0.408248290463863   0.000000000000000  -0.408248290463863  -0.408248290463863  -0.000000000000000   0.408248290463863   0.408248290463863   0.000000000000001  -0.408248290463863
                0.303012985114696  -0.408248290463863  -0.161229841765317   0.464242826880013   0.000000000000000  -0.464242826880013   0.161229841765317   0.408248290463863  -0.303012985114696
                0.161229841765317  -0.408248290463863   0.464242826880013  -0.303012985114695  -0.000000000000000   0.303012985114696  -0.464242826880012   0.408248290463862  -0.161229841765317
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[angW;angU],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,...
                sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec4Ch5Ord0Ang(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 0;
            ang = 2*pi*rand(4,1);
            
            % Expected values
            dimExpctd = [5 4];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            
        end

        % Test for construction
        function testConstructorWithDec4Ch7Ord0Ang(testCase)
            
            % Parameters
            decch = [ 4 7 ];
            ord   = 0;
            ang = 2*pi*rand(9,1);
            
            % Expected values
            dimExpctd = [7 4];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec4Ch9Ord0Ang(testCase)
            
            % Parameters
            decch = [ 4 9 ];
            ord   = 0;
            ang = 2*pi*rand(16,1);
            
            % Expected values
            dimExpctd = [9 4];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec9Ch11Ord0Ang(testCase)
            
            % Parameters
            decch = [ 9 11 ];
            ord   = 0;
            ang = 2*pi*rand(25,1);
            
            % Expected values
            dimExpctd = [11 9];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec1Ch5Ord0(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord   = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [...
                1;
                0;
                0;
                0;
                0];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec1Ch5Ord0Ang0(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord   = 0;
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                1 ;
                0 ;
                0 ;
                0 ;
                0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec1Ch5Ord0AngPi3(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord   = 0;
            angW = zeros(3,1);
            angU = pi/3;
            
            % Expected values
            import saivdr.dictionary.utility.*
            omgsW = OrthonormalMatrixGenerationSystem();
            omgsU = OrthonormalMatrixGenerationSystem();
            matrixW0 = step(omgsW,angW,1);
            matrixU0 = step(omgsU,angU,1);
            coefExpctd(:,:,1) = ...
                blkdiag(matrixW0, matrixU0) * ...
                [ 1 0 0 0 0 ].';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[angW;angU],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Invalid arguments
        function testConstructorWithInvalidArguments(testCase)
            
            % Invalid input
            decch = [ 4 5 ];
            ord   = 0;
            sizeInvalid = [2 2];
            ang = 2*pi*rand(sizeInvalid);
            
            % Expected value
            sizeExpctd = [4 1];
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Size of angles must be [ %d %d ]',...
                sizeExpctd(1), sizeExpctd(2));
            
            % Instantiation of target class
            try
                import saivdr.dictionary.olpprfb.*
                testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                    'DecimationFactor',decch(1),...
                    'NumberOfChannels',decch(2:end),...
                    'PolyPhaseOrder',ord);
                step(testCase.lppufb,ang,[]);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
        end
        
        % Test for construction
        function testConstructorWithMusPosNeg(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord   = 0;
            ang = [ 0 0 0 0 ].';
            mus = [ 1 1 1 -1 -1 ].';
            
            % Expected values
            coefExpctd(:,:,1) = diag(mus)*[
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
            ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 2
        function testConstructorWithDec4Ch5Ord4Ang0(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord   = 4;
            ang   = 0;
            
            % Expected values
            coefExpctd = zeros(5,4,5);
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 8
        function testConstructorWithDec4Ch5Ord8Ang(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord   = 8;
            ang = 2*pi*rand(4,5);
            
            % Expected values
            nDecs = decch(1);
            dimExpctd = [decch(2) nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test dec 9 order 4
        function testConstructorWithDec9Ch9Ord4Ang0(testCase)
            
            % Parameters
            decch = [ 9 9 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(9,9,5);
            coefExpctd(:,:,3) = [
                0.333333333333333   0.333333333333333   0.333333333333333   0.333333333333333   0.333333333333333   0.333333333333333   0.333333333333333   0.333333333333333   0.333333333333333
                0.442975349592450   0.235702260395516  -0.081858535979315  -0.361116813613135  -0.471404520791032  -0.361116813613135  -0.081858535979315   0.235702260395516   0.442975349592450
                0.361116813613135  -0.235702260395516  -0.442975349592450   0.081858535979315   0.471404520791032   0.081858535979316  -0.442975349592450  -0.235702260395516   0.361116813613135
                0.235702260395516  -0.471404520791032   0.235702260395516   0.235702260395516  -0.471404520791032   0.235702260395515   0.235702260395516  -0.471404520791032   0.235702260395516
                0.081858535979315  -0.235702260395516   0.361116813613135  -0.442975349592450   0.471404520791032  -0.442975349592450   0.361116813613135  -0.235702260395515   0.081858535979316
                0.464242826880013   0.408248290463863   0.303012985114696   0.161229841765317   0.000000000000000  -0.161229841765317  -0.303012985114696  -0.408248290463863  -0.464242826880013
                0.408248290463863   0.000000000000000  -0.408248290463863  -0.408248290463863  -0.000000000000000   0.408248290463863   0.408248290463863   0.000000000000001  -0.408248290463863
                0.303012985114696  -0.408248290463863  -0.161229841765317   0.464242826880013   0.000000000000000  -0.464242826880013   0.161229841765317   0.408248290463863  -0.303012985114696
                0.161229841765317  -0.408248290463863   0.464242826880013  -0.303012985114695  -0.000000000000000   0.303012985114696  -0.464242826880012   0.408248290463862  -0.161229841765317
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,...
                sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 order 4
        function testConstructorWithDec4Ch7Ord4Ang(testCase)
            
            % Parameters
            decch = [ 4 7 ];
            ord = 4;
            ang = 2*pi*rand(9,3);
            
            % Expected values
            nDecs = decch(1);
            nChs = decch(2);
            dimExpctd = [nChs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test: dec 9 order 8
        function testConstructorWithDecCh9Ord8Ang(testCase)
            
            % Parameters
            decch = [ 9 9 ];
            ord = 8;
            ang = 2*pi*rand(16,5);
            
            % Expected values
            nDecs = decch(1);
            nChs = decch(2);
            dimExpctd = [nChs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),...
                sprintf('%g',coefDist));
            
        end
        
        % Test for angle setting
        function testSetAngles(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 0;
            angPre = [ pi/4 pi/4 pi/4 pi/4 ].';
            angPst = [ 0 0 0 0 ].';
            
            % Expected values
            coefExpctd = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,angPre,[]);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThanOrEqualTo;
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThanOrEqualTo(1e-14),...
                sprintf('%g',coefDist));
            
            % Actual values
            coefActual = step(testCase.lppufb,angPst,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,...
                sprintf('%g',coefDist));
            
        end
        
        % Test for mus setting
        function testSetMus(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 0;
            ang = [ 0 0 0 0 ].';
            musPre = [ 1 -1  1 -1 1].';
            musPst = 1;
            
            % Expected values
            coefExpctd = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,musPre);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThanOrEqualTo;
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThanOrEqualTo(1e-14),sprintf('%g',coefDist));
            
            % Actual values
            coefActual = step(testCase.lppufb,[],musPst);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,...
                sprintf('%g',coefDist));
            
        end
        
        % Test for subsref
        function testAnalysisFilterAt(testCase)
            
            % Expected value
            anFiltExpctd1 = [ 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000 ];
            anFiltExpctd2 = [ 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000 ];
            anFiltExpctd3 = [0 0 0 0 ];
            anFiltExpctd4 = [ 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188 ];
            anFiltExpctd5 = [ 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'OutputMode','AnalysisFilterAt');
            
            % Actual values
            anFiltActual1 = step(testCase.lppufb,[],[],1);
            anFiltActual2 = step(testCase.lppufb,[],[],2);
            anFiltActual3 = step(testCase.lppufb,[],[],3);
            anFiltActual4 = step(testCase.lppufb,[],[],4);
            anFiltActual5 = step(testCase.lppufb,[],[],5);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            dist = norm(anFiltExpctd1(:)-anFiltActual1(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd2(:)-anFiltActual2(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd3(:)-anFiltActual3(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd4(:)-anFiltActual4(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd5(:)-anFiltActual5(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            
        end
           
        function testAnalysisFilters(testCase)
            
            % Expected value
            anFiltExpctd1 = [ 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000 ];
            anFiltExpctd2 = [ 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000 ];
            anFiltExpctd3 = [ 0 0 0 0 ];
            anFiltExpctd4 = [ 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188 ];
            anFiltExpctd5 = [ 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'OutputMode','AnalysisFilters');
            
            % Actual values
            anFiltsActual = step(testCase.lppufb,[],[]);
            anFiltActual1 = anFiltsActual(:,1);
            anFiltActual2 = anFiltsActual(:,2);
            anFiltActual3 = anFiltsActual(:,3);
            anFiltActual4 = anFiltsActual(:,4);
            anFiltActual5 = anFiltsActual(:,5);            
            
           % Evaluation
            import matlab.unittest.constraints.IsLessThan
            dist = norm(anFiltExpctd1(:)-anFiltActual1(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd2(:)-anFiltActual2(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd3(:)-anFiltActual3(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd4(:)-anFiltActual4(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd5(:)-anFiltActual5(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            
        end 

        % Test dec 4 ch 5 order 2
        function testConstructorWithDec4Ch5Ord2(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 2;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(5,4,3);
            coefExpctd(:,:,2) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 ch 5 order 2
        function testConstructorWithDec1Ch5Ord2(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord = 2;
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0  ;
                0  ;
                0  ;
                0  ;
                0  ];
            
            coefExpctd(:,:,2) = [
                1 ;
                0 ;
                0 ;
                0 ;
                0 ];
            
            coefExpctd(:,:,3) = [
                0  ;
                0  ;
                0  ;
                0  ;
                0  ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 ch 5 order 4
        function testConstructorWithDec1Ch5Ord4(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2) =  [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3) = [...
                1;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,4) =  [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,5) = [...
                0;
                0;
                0;
                0;
                0];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 4
        function testConstructorWithDec4Ch5Ord4(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(5,4,5);
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 order 8
        function testConstructorWithDec4Ch5Ord8(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 8;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(5,4,9);
            coefExpctd(:,:,5) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
         % Test dec 4 ch 5 order 12
        function testConstructorWithDec4Ch5Ord12Ang(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 12;
            ang = 2*pi*rand(4,7);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test dec 4 ch 9 order 4
        function testConstructorWithDec4Ch9Ord4Ang(testCase)
            
            % Parameters
            decch = [ 4 9 ];
            ord = 4;
            ang = 2*pi*rand(16,3);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 ch 9 order 8
        function testConstructorWithDec4Ch9Ord8Ang(testCase)
            
            % Parameters
            decch = [ 4 9 ];
            ord = 8;
            ang = 2*pi*rand(16,5);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 ch 1 order 4
        function testConstructorWithDec4Ch11Ord4Ang(testCase)
            
            % Parameters
            decch = [ 4 11 ];
            ord = 4;
            ang = 2*pi*rand(25,3);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 6 ch 11 order 4
        function testConstructorWithDec6Ch11Ord4Ang(testCase)
            
            % Parameters
            decch = [ 6 11 ];
            ord = 4;
            ang = 2*pi*rand(25,3);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);            
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 6 ch 11 order 8
        function testConstructorWithDec6Ch11Ord8Ang(testCase)
            
            % Parameters
            decch = [ 6 11 ];
            ord = 8;
            ang = 2*pi*rand(25,5);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
                        
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 ch 5 order 4
        function testConstructorWithDec1Ch5Ord4Ang(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord = 4;
            ang = 2*pi*rand(4,3);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
                        
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 ch 5 order 8
        function testConstructorWithDec1Ch5Ord8Ang(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord = 8;
            ang = 2*pi*rand(4,5);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 ch 7 order 8
        function testConstructorWithDec1Ch7Ord8Ang(testCase)
            
            % Parameters
            decch = [ 1 7 ];
            ord = 8;
            ang = 2*pi*rand(9,5);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 ch 7 order 12
        function testConstructorWithDec1Ch7Ord12Ang(testCase)
            
            % Parameters
            decch = [ 1 7 ];
            ord = 12;
            ang = 2*pi*rand(9,7);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec4Ch32Ord0(testCase)
            
            % Parameters
            decch = [ 4 3 2 ];
            ord = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0 
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec4Ch42Ord0(testCase)
            
            % Parameters
            decch = [ 4 4 2 ];
            ord = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec4Ch43Ord0(testCase)
            
            % Parameters
            decch = [ 4 4 3 ];
            ord = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
                0 0 0 0                
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec4Ch52Ord0(testCase)
            
            % Parameters
            decch = [ 4 5 2 ];
            ord = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0 0 0 0
                0 0 0 0                
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec4Ch62Ord0(testCase)
            
            % Parameters
            decch = [ 4 6 2 ];
            ord = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0 0 0 0
                0 0 0 0                
                0 0 0 0                        
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 4
        function testConstructorWithDec4Ch32Ord4(testCase)
            
            % Parameters
            decch = [ 4 3 2 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd  = zeros(5,4,5);
            
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4
        function testConstructorWithDec4Ch42Ord4(testCase)
            
            % Parameters
            decch = [ 4 4 2 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(6,4,5);
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4
        function testConstructorWithDec4Ch52Ord4(testCase)
            
            % Parameters
            decch = [ 4 5 2 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(7,4,5);
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0 0 0 0
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4
        function testConstructorWithDec4Ch53Ord4(testCase)
            
            % Parameters
            decch = [ 4 5 3 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(8,4,5);
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0 0 0 0
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
                0 0 0 0
                ];            

            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 4
        function testConstructorWithDec4Ch62Ord4(testCase)
            
            % Parameters
            decch = [ 4 6 2 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(8,4,5);
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0 0 0 0
                0 0 0 0
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
                ];            
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test dec 4 order 8
        function testConstructorWithDec4Ch32Ord8(testCase)
            
            % Parameters
            decch = [ 4 3 2 ];
            ord = 8;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(5,4,9);
            coefExpctd(:,:,5) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
                ];            
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end     

        % Test dec 4 order 8
        function testConstructorWithDec4Ch42Ord8(testCase)
            
            % Parameters
            decch = [ 4 4 2 ];
            ord = 8;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(6,4,9);
            coefExpctd(:,:,5) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0
                0 0 0 0
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
                ];                        
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end  
        
        % Test dec 4 order 4
        function testConstructorWithDec4Ch32Ord4Ang(testCase)
            
          % Parameters
            decch = [ 4 3 2 ];
            ord = 4;
            ang = 2*pi*rand(4,3);
            
            % Expected values
            nDec = prod(decch(1));
            nChs = sum(decch(2:end));
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 order 8
        function testConstructorWithDec4Ch32Ord8Ang(testCase)
            
            % Parameters
            decch = [ 4 3 2 ];
            ord = 8;
            ang = 2*pi*rand(4,5);
            
            % Expected values
            nDecs = decch(1);
            nChs = sum(decch(2:end));
            dimExpctd = [nChs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 order 4 
        function testConstructorWithDec4Ch42Ord4Ang(testCase)
            
          % Parameters
            decch = [ 4 4 2 ];
            ord = 4;
            ang = 2*pi*rand(7,3);
            
            % Expected values
            nDec = prod(decch(1));
            nChs = sum(decch(2:end));
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),sprintf('%g',coefDist));
            coefOdd = coefActual(decch(2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 order 8
        function testConstructorWithDec4Ch42Ord8Ang(testCase)
            
            % Parameters
            decch = [ 4 4 2 ];
            ord = 8;
            ang = 2*pi*rand(7,5);
            
            % Expected values
            nDecs = decch(1);
            nChs = sum(decch(2:end));
            dimExpctd = [nChs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),sprintf('%g',coefDist));
            coefOdd = coefActual(decch(2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),sprintf('%g',coefDist));
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
       
        % Test for ParameterMatrixSet
        function testParameterMatrixContainer(testCase)
            
            % Preparation
            mstab = [ 3 3 ; 2 2 ];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixContainer
            paramExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(3),1);
            step(paramExpctd,eye(2),2);
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end
        
        % Test for construction
        function testConstructorWithDec4Ch24Ord0(testCase)
            
            % Parameters
            decch = [ 4 2 4 ];
            ord = 0;
            
            % Expected values
            coefExpctd = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                0 0 0 0
                0 0 0 0
                ];
        
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec4Ch34Ord0(testCase)
            
            % Parameters
            decch = [ 4 3 4 ];
            ord = 0;
            
            % Expected values
            coefExpctd = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0                
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                0 0 0 0
                0 0 0 0
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec4Ch25Ord0(testCase)
            
            % Parameters
            decch = [ 4 2 5 ];
            ord = 0;
            
            % Expected values
            coefExpctd = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                0 0 0 0
                0 0 0 0 
                0 0 0 0 
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec4Ch26Ord0(testCase)
            
            % Parameters
            decch = [ 4 2 6 ];
            ord = 0;
            
            % Expected values
            coefExpctd = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                0 0 0 0 
                0 0 0 0 
                0 0 0 0 
                0 0 0 0
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4
        function testConstructorWithDec4Ch23Ord4(testCase)
            
            % Parameters
            decch = [ 4 2 3 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(5,4,5);
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                0 0 0 0 
                ];

            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 4
        function testConstructorWithDec4Ch24Ord4(testCase)
            
            % Parameters
            decch = [ 4 2 4 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(6,4,5);
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                0 0 0 0 
                0 0 0 0                 
                ];

            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 4
        function testConstructorWithDec4Ch25Ord4(testCase)
            
            % Parameters
            decch = [ 4 2 5 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(7,4,5);
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                0 0 0 0 
                0 0 0 0 
                0 0 0 0                 
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4
        function testConstructorWithDec4Ch35Ord4(testCase)
            
            % Parameters
            decch = [ 4 3 5 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(8,4,5);
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0 0 0 0 
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                0 0 0 0 
                0 0 0 0 
                0 0 0 0                 
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 4
        function testConstructorWithDec4Ch26Ord4(testCase)
            
            % Parameters
            decch = [ 4 2 6 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(8,4,5);
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                0 0 0 0 
                0 0 0 0 
                0 0 0 0 
                0 0 0 0                 
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test dec 4 order 8
        function testConstructorWithDec4Ch23Ord8(testCase)
            
            % Parameters
            decch = [ 4 2 3 ];
            ord = 8;  
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(5,4,9);
            coefExpctd(:,:,5) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                0 0 0 0                 
                ];

            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end     

        % Test dec 4 order 8
        function testConstructorWithDec4Ch24Ord8(testCase)
            
            % Parameters
            decch = [ 4 2 4 ];
            ord = 8;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(6,4,9);
            coefExpctd(:,:,5) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                0 0 0 0                 
                0 0 0 0                 
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end  
        
        % Test dec 4 order 4
        function testConstructorWithDec4Ch23Ord4Ang(testCase)
            
          % Parameters
            decch = [ 4 2 3 ];
            ord = 4;
            ang = 2*pi*rand(4,3);
            
            % Expected values
            nDec = prod(decch(1));
            nChs = sum(decch(2:end));
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:floor(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(floor(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test dec 4 order 8
        function testConstructorWithDec4Ch23Ord8Ang(testCase)
            
            % Parameters
            decch = [ 4 2 3 ];
            ord = 8;
            ang = 2*pi*rand(4,5);
            
            % Expected values
            nDecs = decch(1);
            nChs = sum(decch(2:end));
            dimExpctd = [nChs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:floor(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(floor(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test dec 4 order 4
        function testConstructorWithDec4Ch24Ord4Ang(testCase)
            
          % Parameters
            decch = [ 4 2 4 ];
            ord = 4;
            ang = 2*pi*rand(7,3);
            
            % Expected values
            nDec = prod(decch(1));
            nChs = sum(decch(2:end));
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),sprintf('%g',coefDist));
            coefOdd = coefActual(decch(2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 order 8
        function testConstructorWithDec4Ch24Ord8Ang(testCase)
            
            % Parameters
            decch = [ 4 2 4 ];
            ord = 8;
            ang = 2*pi*rand(7,5);
            
            % Expected values
            nDecs = decch(1);
            nChs = sum(decch(2:end));
            dimExpctd = [nChs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),sprintf('%g',coefDist));
            coefOdd = coefActual(decch(2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),sprintf('%g',coefDist));
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
       
        function testParameterMatrixSetCh23(testCase)
            
            % Preparation
            nchs  = [ 2 3 ];
            mstab = [ 2 2 ; 3 3 ];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixContainer
            paramExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(2),1);
            step(paramExpctd,eye(3),2);
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm0System(...
                'OutputMode','ParameterMatrixSet',...
                'NumberOfChannels',nchs);
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end

    end
    
end
