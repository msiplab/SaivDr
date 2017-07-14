classdef OvsdLpPuFb1dTypeIVm1SystemTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB1dTYPEIVM1TESTCASE Test case for OvsdLpPuFb1dTypeIVm1System
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
        lppufb
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
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System();
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,...
                sprintf('%g',coefDist));
            
        end
        
        % Test for default construction
        function testConstructorWithDeepCopy(testCase)
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System();
            cloneLpPuFb = clone(testCase.lppufb);

            % Evaluation
            testCase.verifyEqual(cloneLpPuFb,testCase.lppufb);
            testCase.verifyFalse(cloneLpPuFb == testCase.lppufb);
            prpOrg = get(testCase.lppufb,'ParameterMatrixSet');
            prpCln = get(cloneLpPuFb,'ParameterMatrixSet');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            
            % Expected values
            coefExpctd = step(testCase.lppufb,[],[]);
            
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

        % Test for construction
        function testConstructorWithOrd0(testCase)
            
            % Parameters
            dec = 4;
            ord = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,...
                sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec8Ord0(testCase)
            
            % Parameters
            dec = 8;
            ord = 0;
            
            % Expected values
            dimExpctd = [8 8];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
        end
        

        % Test for construction
        function testConstructorWithDe4Ch4Ord0(testCase)
            
            % Parameters
            decch = [ 4 4 ];
            ord = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099    
            ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,...
                sprintf('%g',coefDist));

        end

         % Test for construction with order 2
        function testConstructorWithDec4Ch6Ord2(testCase)
            
            % Parameters
            decch = [ 4 6 ];
            ord = 2;
            ang = 2*pi*rand(3,4);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1]; 
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
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
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyTrue(coefDist<1e-15,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 2
        function testConstructorWithDec4Ch8Ord2(testCase)
            
            % Parameters
            decch = [ 4 8 ];
            ord = 2;
            ang = 2*pi*rand(6,4);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
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
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyTrue(coefDist<1e-15,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec4Ch8Ord0(testCase)
            
            % Parameters
            decch = [ 4 8 ];
            ord = 0;
            
            % Expected values
            dimExpctd = [8 4];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
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
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyTrue(coefDist<1e-15,sprintf('%g',coefDist));
            
        end
       
        % Test for construction
        function testConstructorWithDec4Ch6Ord0(testCase)
            
            % Parameters
            decch = [ 4 6 ];
            ord = 0;
            
            % Expected values
            dimExpctd = [6 4];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
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
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyTrue(coefDist<1e-15,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithOrd0Ang(testCase)
            
            % Parameters
            dec = 4;
            ord = 0;
            ang = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./...
                (abs(coefExpctd(:))));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,...
                sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithAng0Pi4(testCase)
            
            % Parameters
            dec = 4;
            ord = 0;
            ang = [ 0 pi/4 ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            omgs = OrthonormalMatrixGenerationSystem();
            matrixW0 = step(omgs,ang(1),1);
            matrixU0 = step(omgs,ang(2),1);
            coefExpctd(:,:,1) = ...
                blkdiag(matrixW0, matrixU0) * [...
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-15,...
                sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec8Ord0Ang(testCase)
            
            % Parameters
            dec = 8;
            ord = 0;
            ang = 2*pi*rand(6,2);
            
            % Expected values
            dimExpctd = [8 8];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
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
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check orthogonality
            import matlab.unittest.constraints.IsLessThan
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec4Ch8Ord0Ang(testCase)
            
            % Parameters
            decch = [ 4 8 ];
            ord = 0;
            ang = 2*pi*rand(6,2);
            
            % Expected values
            dimExpctd = [8 4];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
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
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check orthogonality
            import matlab.unittest.constraints.IsLessThan
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 ch 6 order 4
        function testConstructorWithDec4Ch6Ord4AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 4 6 ];
            ord = 4;
            ang = 2*pi*rand(3,6);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*            
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check DC-E
            release(testCase.lppufb)
            import matlab.unittest.constraints.IsLessThan
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            for iSubband = 2:nChs
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
        end
        
        % Test dec 2 2 ch 6 order 2 2
        function testConstructorWithDec4Ch6Ord8AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 4 6 ];
            ord = 8;
            ang = 2*pi*rand(3,10);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*            
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check DC-leakage
            release(testCase.lppufb)
            import matlab.unittest.constraints.IsLessThan
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            for iSubband = 2:nChs
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
        end

        % Test dec 2 2 ch 6 order 2 2
        function testConstructorWithDec4Ch8Ord8AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 4 4 4 ];
            ord = 8;
            ang = 2*pi*rand(6,10);
            
            % Expected values
            nChs = sum(decch(2:3));
            nDec = prod(decch(1));
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*            
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check DC-leakage
            release(testCase.lppufb)
            import matlab.unittest.constraints.IsLessThan
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            for iSubband = 2:nChs
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
        end
        
        % Test for ParameterMatrixSet
        function testParameterMatrixContainer(testCase)
            
            % Preparation
            mstab = [ 2 2 ; 2 2 ];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixContainer
            paramExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(2),1);
            step(paramExpctd,eye(2),2);
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end
        
        % Test for construction with order 4
        function testParameterMatrixSetRandAngMuWithDec4Ch22Ord4(testCase)
            
            % Parameters
            decch = [ 4 2 2 ];
            ord = 4;
            mstab = [ 2 2 ; 2 2 ; 2 2 ; 2 2 ; 2 2 ; 2 2 ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
            step(paramMtxExpctd,-eye(mstab(3,:)),uint32(3)); % U1
            step(paramMtxExpctd,-eye(mstab(4,:)),uint32(4)); % U2
            step(paramMtxExpctd,-eye(mstab(5,:)),uint32(5)); % U3
            step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % U4
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramMtxActual = step(testCase.lppufb,[],[]);
            coefActual = get(paramMtxActual,'Coefficients');
            
            % Evaluation
            diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
            
            %
            ang = get(testCase.lppufb,'Angles');
            ang = randn(size(ang));
            mus = get(testCase.lppufb,'Mus');
            mus = 2*(rand(size(mus))>0.5)-1;
            %
            import saivdr.dictionary.utility.*
            omgs = OrthonormalMatrixGenerationSystem();
            W0  = step(omgs,0     , [1; mus(2,1)]);
            U0  = step(omgs,ang(2), mus(:,2));
            U1 = step(omgs,ang(3), mus(:,3));
            U2 = step(omgs,ang(4), mus(:,4));
            U3 = step(omgs,ang(5), mus(:,5));
            U4 = step(omgs,ang(6), mus(:,6));
            step(paramMtxExpctd,W0 ,uint32(1)); % W0
            step(paramMtxExpctd,U0 ,uint32(2)); % U0
            step(paramMtxExpctd,U1,uint32(3));  % U1
            step(paramMtxExpctd,U2,uint32(4));  % U2
            step(paramMtxExpctd,U3,uint32(5));  % U3            
            step(paramMtxExpctd,U4,uint32(6));  % U4
            %
            coefExpctd = get(paramMtxExpctd,'Coefficients');            
            
            %
            set(testCase.lppufb,'Angles',ang,'Mus',mus);

            % Actual values
            paramMtxActual = step(testCase.lppufb,ang,mus);
            coefActual = get(paramMtxActual,'Coefficients');
            
            % Evaluation
            diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
            
            % Check DC-E
            release(testCase.lppufb)
            import matlab.unittest.constraints.IsLessThan
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            for iSubband = 2:sum(decch(2:3))
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
        end
        
        % Test for construction with order 4
        function testParameterMatrixSetRandAngWithDec4Ch22Ord4(testCase)
            
            % Parameters
            decch = [ 4 2 2 ];
            ord = 4;
            mstab = [ 2 2 ; 2 2 ; 2 2 ; 2 2 ; 2 2 ; 2 2 ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
            step(paramMtxExpctd,-eye(mstab(3,:)),uint32(3)); % U1
            step(paramMtxExpctd,-eye(mstab(4,:)),uint32(4)); % U2
            step(paramMtxExpctd,-eye(mstab(5,:)),uint32(5)); % U3
            step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % U4
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramMtxActual = step(testCase.lppufb,[],[]);
            coefActual = get(paramMtxActual,'Coefficients');
            
            % Evaluation
            diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
            
            %
            ang = get(testCase.lppufb,'Angles');
            ang = randn(size(ang));
            mus = get(testCase.lppufb,'Mus');
            %
            import saivdr.dictionary.utility.*
            omgs = OrthonormalMatrixGenerationSystem();
            W0  = step(omgs,0     , [1; mus(2,1)]);
            U0  = step(omgs,ang(2), mus(:,2));
            U1 = step(omgs,ang(3), mus(:,3));
            U2 = step(omgs,ang(4), mus(:,4));
            U3 = step(omgs,ang(5), mus(:,5));
            U4 = step(omgs,ang(6), mus(:,6));
            step(paramMtxExpctd,W0 ,uint32(1)); % W0
            step(paramMtxExpctd,U0 ,uint32(2)); % U0
            step(paramMtxExpctd,U1 ,uint32(3)); % U1
            step(paramMtxExpctd,U2 ,uint32(4)); % U2
            step(paramMtxExpctd,U3 ,uint32(5)); % U3            
            step(paramMtxExpctd,U4 ,uint32(6));  % U4
            %
            coefExpctd = get(paramMtxExpctd,'Coefficients');            
            
            %
            set(testCase.lppufb,'Angles',ang);

            % Actual values
            paramMtxActual = step(testCase.lppufb,ang,mus);
            coefActual = get(paramMtxActual,'Coefficients');
            
            % Evaluation
            diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
            
            % Check DC-E
            release(testCase.lppufb)
            import matlab.unittest.constraints.IsLessThan
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            for iSubband = 2:sum(decch(2:3))
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end            
        end  

        % Test for construction with order 4
        function testParameterMatrixSetRandMuWithDec4Ch22Ord4(testCase)
            
            % Parameters
            decch = [ 4 2 2 ];
            ord = 4;
            mstab = [ 2 2 ; 2 2 ; 2 2 ; 2 2 ; 2 2 ; 2 2 ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
            step(paramMtxExpctd,-eye(mstab(3,:)),uint32(3)); % U1
            step(paramMtxExpctd,-eye(mstab(4,:)),uint32(4)); % U2
            step(paramMtxExpctd,-eye(mstab(5,:)),uint32(5)); % U3
            step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % U4
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramMtxActual = step(testCase.lppufb,[],[]);
            coefActual = get(paramMtxActual,'Coefficients');
            
            % Evaluation
            diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
            
            %
            ang = get(testCase.lppufb,'Angles');
            mus = get(testCase.lppufb,'Mus');
            mus = 2*(rand(size(mus))>0.5)-1;
            %
            import saivdr.dictionary.utility.*
            omgs = OrthonormalMatrixGenerationSystem();
            W0 = step(omgs,0     , [1; mus(2,1)]);
            U0 = step(omgs,ang(2), mus(:,2));
            U1 = step(omgs,ang(3), mus(:,3));
            U2 = step(omgs,ang(4), mus(:,4));
            U3 = step(omgs,ang(5), mus(:,5));
            U4 = step(omgs,ang(6), mus(:,6));
            step(paramMtxExpctd,W0,uint32(1)); % W0
            step(paramMtxExpctd,U0,uint32(2)); % U0
            step(paramMtxExpctd,U1,uint32(3)); % U1
            step(paramMtxExpctd,U2,uint32(4)); % U2
            step(paramMtxExpctd,U3,uint32(5)); % U3            
            step(paramMtxExpctd,U4,uint32(6)); % U4
            %
            coefExpctd = get(paramMtxExpctd,'Coefficients');            
            
            %
            set(testCase.lppufb,'Mus',mus);

            % Actual values
            paramMtxActual = step(testCase.lppufb,ang,mus);
            coefActual = get(paramMtxActual,'Coefficients');
            
            % Evaluation
            diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
            
            % Check DC-E
            release(testCase.lppufb)
            import matlab.unittest.constraints.IsLessThan
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            for iSubband = 2:sum(decch(2:3))
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
        end  

    end
    
end
