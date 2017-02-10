classdef CplxOvsdLpPuFb1dTypeIVm1SystemTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB1dTYPEIVM1TESTCASE Test case for CplxOvsdLpPuFb1dTypeIVm1System
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
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
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i,...
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i, -0.353553390593274 - 0.353553390593274i,...
                -0.353553390593274 + 0.353553390593274i,  0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i,  0.000000000000000 + 0.500000000000000i,...
                 0.000000000000000 - 0.500000000000000i,  0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i,  0.353553390593274 - 0.353553390593274i,...
                 0.353553390593274 + 0.353553390593274i, -0.353553390593274 + 0.353553390593274i];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System();
            
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
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System();
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
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i,...
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i, -0.353553390593274 - 0.353553390593274i,...
                -0.353553390593274 + 0.353553390593274i,  0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i,  0.000000000000000 + 0.500000000000000i,...
                 0.000000000000000 - 0.500000000000000i,  0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i,  0.353553390593274 - 0.353553390593274i,...
                 0.353553390593274 + 0.353553390593274i, -0.353553390593274 + 0.353553390593274i];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
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
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);

            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));     
            
            % Check orthogonality
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd))...
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
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i,...
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i, -0.353553390593274 - 0.353553390593274i,...
                -0.353553390593274 + 0.353553390593274i,  0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i,  0.000000000000000 + 0.500000000000000i,...
                 0.000000000000000 - 0.500000000000000i,  0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i,  0.353553390593274 - 0.353553390593274i,...
                 0.353553390593274 + 0.353553390593274i, -0.353553390593274 + 0.353553390593274i];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
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
            ang = 2*pi*rand(15+12+2,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1]; 
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);

            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));     
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E'*E);
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
            ang = 2*pi*rand(28+24+4,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);

            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));     
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E'*E);
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
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);

            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));     
            
            % Check orthogonality
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))...
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
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);

            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));     
            
            % Check orthogonality
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyTrue(coefDist<1e-15,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithOrd0Ang(testCase)
            
            % Parameters
            dec = 4;
            ord = 0;
            ang = [ 0 0 0 0 0 0 ].';
            
            % Expected values
            coefExpctd(:,:,1) = [
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i,...
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i, -0.353553390593274 - 0.353553390593274i,...
                -0.353553390593274 + 0.353553390593274i,  0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i,  0.000000000000000 + 0.500000000000000i,...
                 0.000000000000000 - 0.500000000000000i,  0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i,  0.353553390593274 - 0.353553390593274i,...
                 0.353553390593274 + 0.353553390593274i, -0.353553390593274 + 0.353553390593274i];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
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
        
        % TODO: テストの妥当性を考える
        % Test for construction
        function testConstructorWithAng0Pi4(testCase)
            
            % Parameters
            dec = 4;
            ord = 0;
            ang = [ 0 0 0 0 0 pi/4 ].';
            
            % Expected values
            import saivdr.dictionary.utility.*
            omgs = OrthonormalMatrixGenerationSystem();
            matrixV0 = step(omgs,ang,1);
            coefExpctd(:,:,1) = ...
                matrixV0 * [
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i,...
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i, -0.353553390593274 - 0.353553390593274i,...
                -0.353553390593274 + 0.353553390593274i,  0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i,  0.000000000000000 + 0.500000000000000i,...
                 0.000000000000000 - 0.500000000000000i,  0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i,  0.353553390593274 - 0.353553390593274i,...
                 0.353553390593274 + 0.353553390593274i, -0.353553390593274 + 0.353553390593274i];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
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
            ang = 2*pi*rand(28,1);
            
            % Expected values
            dimExpctd = [8 8];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);

            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));     
            
            % Check orthogonality
            import matlab.unittest.constraints.IsLessThan
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec4Ch8Ord0Ang(testCase)
            
            % Parameters
            decch = [ 4 8 ];
            ord = 0;
            ang = 2*pi*rand(28,1);
            
            % Expected values
            dimExpctd = [8 4];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);

            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));     
            
            % Check orthogonality
            import matlab.unittest.constraints.IsLessThan
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 ch 6 order 4
        function testConstructorWithDec4Ch6Ord4AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 4 6 ];
            ord = 4;
            ang = 2*pi*rand(15+24+4,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*            
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
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
            ang = 2*pi*rand(15+48+8,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*            
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
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
            ang = 2*pi*rand(28+12*8+16,1);
            
            % Expected values
            nChs = sum(decch(2:3));
            nDec = prod(decch(1));
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*            
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
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
            mstab = [4 4];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixContainer
            paramExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(4),1);
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm0System(...
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
            mstab = [ 4 4 ; 2 2 ; 2 2 ; 1 1 ; 2 2 ; 2 2 ; 1 1 ; 2 2 ; 2 2 ; 1 1 ; 2 2 ; 2 2 ; 1 1;];
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab( 1,:)),uint32( 1)); % V0
            step(paramMtxExpctd, eye(mstab( 2,:)),uint32( 2)); % W1
            step(paramMtxExpctd, eye(mstab( 3,:)),uint32( 3)); % U1
            step(paramMtxExpctd,            0 ,uint32( 4)); % angB1
            step(paramMtxExpctd, eye(mstab( 5,:)),uint32( 5)); % W2
            step(paramMtxExpctd, eye(mstab( 6,:)),uint32( 6)); % U2
            step(paramMtxExpctd,            0 ,uint32( 7)); % angB2
            step(paramMtxExpctd, eye(mstab( 8,:)),uint32( 8)); % W3
            step(paramMtxExpctd, eye(mstab( 9,:)),uint32( 9)); % U3
            step(paramMtxExpctd,            0 ,uint32(10)); % angB3
            step(paramMtxExpctd, eye(mstab(11,:)),uint32(11)); % W4
            step(paramMtxExpctd, eye(mstab(12,:)),uint32(12)); % U4
            step(paramMtxExpctd,            0 ,uint32(13)); % angB4
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
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
            %mus = 2*(rand(size(mus))>0.5)-1;
            mus = ones(size(mus));
            %
            %TODO: musを考慮した設計にする．
            import saivdr.dictionary.utility.*
            initOmgs = OrthonormalMatrixGenerationSystem();
            propOmgs = OrthonormalMatrixGenerationSystem();
            V0 = step(initOmgs, ang(1:6) , [1;1;1;1]);
            W1 = step(propOmgs,ang(7) , [1;1]);
            U1 = step(propOmgs,ang(8), [1;1]);
            angB1 = ang(9);
            W2 = step(propOmgs,ang(10), [1;1]);
            U2 = step(propOmgs,ang(11), [1;1]);
            angB2 = ang(12);
            W3 = step(propOmgs,ang(13), [1;1]);
            U3 = step(propOmgs,ang(14), [1;1]);
            angB3 = ang(15);
            W4 = step(propOmgs,ang(16), [1;1]);
            U4 = step(propOmgs,ang(17), [1;1]);
            angB4 = ang(18);
            
            step(paramMtxExpctd,V0,uint32(1)); % V0
            step(paramMtxExpctd,W1,uint32(2)); % W1
            step(paramMtxExpctd,U1,uint32(3)); % U1
            step(paramMtxExpctd,angB1,uint32(4)); % angB1
            step(paramMtxExpctd,W2,uint32(5));  % W2
            step(paramMtxExpctd,U2,uint32(6));  % U2
            step(paramMtxExpctd,angB2,uint32(7)); % angB2
            step(paramMtxExpctd,W3,uint32(8));  % W3            
            step(paramMtxExpctd,U3,uint32(9));  % U3
            step(paramMtxExpctd,angB3,uint32(10)); % angB1
            step(paramMtxExpctd,W4,uint32(11));  % W4            
            step(paramMtxExpctd,U4,uint32(12));  % U4
            step(paramMtxExpctd,angB4,uint32(13)); % angB1
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
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
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
            mstab = [ 4 4 ; 
                2 2 ; 2 2 ; 1 1 ;
                2 2 ; 2 2 ; 1 1 ;
                2 2 ; 2 2 ; 1 1 ;
                2 2 ; 2 2 ; 1 1 ; ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % V0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % W1
            step(paramMtxExpctd,-eye(mstab(3,:)),uint32(3)); % U1
            step(paramMtxExpctd, pi/4*ones(mstab(4,:)),uint32(4)); % angB1
            step(paramMtxExpctd, eye(mstab(5,:)),uint32(5)); % W2
            step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % U2
            step(paramMtxExpctd, pi/4*ones(mstab(3,:)),uint32(7)); % angB2
            step(paramMtxExpctd, eye(mstab(8,:)),uint32(8)); % W3
            step(paramMtxExpctd,-eye(mstab(9,:)),uint32(9)); % U3
            step(paramMtxExpctd, pi/4*(mstab(10,:)),uint32(10)); % angB3
            step(paramMtxExpctd, eye(mstab(11,:)),uint32(11)); % W3
            step(paramMtxExpctd,-eye(mstab(12,:)),uint32(12)); % U3
            step(paramMtxExpctd, pi/4*(mstab(13,:)),uint32(13)); % angB3
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIVm1System(...
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
