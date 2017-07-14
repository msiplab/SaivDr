classdef OvsdLpPuFb1dTypeIIVm1SystemTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB1dTYPEIIVM1SYSTEMTESTCASE Test case for OvsdLpPuFb1dTypeIIVm1System
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
                0 0 0 0 
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System();
            
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
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System();
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
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,...
                sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneLpPuFb,'Angles')));

            % Actual values
            coefActual = step(cloneLpPuFb,angles,[]);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThan(1e-14),...
                sprintf('%g',coefDist));
            
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
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System(...
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
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System(...
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
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System(...
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
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System(...
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
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System(...
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
        function testConstructorWithDec4Ch5Ord4AngNoDcLeakage(testCase)
            
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
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System(...
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
            for iSubband = 2:nChs;
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
        end

        % Test for construction with order 4
        function testConstructorWithDec4Ch5Ord8AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 8;
            ang = 2*pi*rand(4,5);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System(...
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
            for iSubband = 2:nChs;
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
        end
        
        % Test for construction with order 4
        function testConstructorWithDec4Ch42Ord8AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 4 4 2 ];
            ord = 8;
            ang = 2*pi*rand(7,5);
            
            % Expected values
            nChs = sum(decch(2:end));
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System(...
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
            for iSubband = 2:nChs;
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
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
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end
        
        % Test for construction with order 4
        function testParameterMatrixSetRandAngWithDec4Ch32Ord4(testCase)
            
            % Parameters
            decch = [ 4 3 2 ];
            ord = 4;
            mstab = [ 3 3 ; 2 2 ; 3 3 ; 2 2 ; 3 3 ; 2 2 ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
            step(paramMtxExpctd, eye(mstab(3,:)),uint32(3)); % W1
            step(paramMtxExpctd,-eye(mstab(4,:)),uint32(4)); % U1
            step(paramMtxExpctd, eye(mstab(5,:)),uint32(5)); % W2
            step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % U2
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System(...
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
            
            % Random angles
            ang = get(testCase.lppufb,'Angles');
            ang = randn(size(ang));
            
            % Expected vales
            coefExpctd = 1;
            
            % Actual values
            set(testCase.lppufb,'Angles',ang);
            paramMtxActual = step(testCase.lppufb,ang,[]);
            W0 = step(paramMtxActual,[],uint32(1));
            W1 = step(paramMtxActual,[],uint32(3));
            W2 = step(paramMtxActual,[],uint32(5));
            G = W2*W1*W0;
            coefActual = G(1,1);

            % Evaluation
            diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
            
            % Check DC-E
            release(testCase.lppufb)
            import matlab.unittest.constraints.IsLessThan
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            for iSubband = 2:sum(decch(2:end))
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
        end

        % Test for construction with order 4
        function testParameterMatrixSetRandMusWithDec4Ch23Ord4(testCase)
            
            % Parameters
            decch = [ 4 2 3 ];
            ord = 4;
            mstab = [ 2 2; 3 3 ; 2 2 ; 3 3 ; 2 2 ; 3 3 ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
            step(paramMtxExpctd,-eye(mstab(3,:)),uint32(3)); % W1
            step(paramMtxExpctd, eye(mstab(4,:)),uint32(4)); % U1
            step(paramMtxExpctd,-eye(mstab(5,:)),uint32(5)); % W2
            step(paramMtxExpctd, eye(mstab(6,:)),uint32(6)); % U2
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System(...
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
            
            % Random angles
            mus = get(testCase.lppufb,'Mus');
            mus = 2*(rand(size(mus))>0.5)-1;
            
            % Expected vales
            coefExpctd = 1;
            
            % Actual values
            set(testCase.lppufb,'Mus',mus);
            paramMtxActual = step(testCase.lppufb,[],mus);
            W0  = step(paramMtxActual,[],uint32(1));
            W1 = step(paramMtxActual,[],uint32(3));
            W2 = step(paramMtxActual,[],uint32(5));
            G = W2*W1*W0;
            coefActual = G(1,1);
            
            % Evaluation
            diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
            
            % Check DC-E
            release(testCase.lppufb)
            import matlab.unittest.constraints.IsLessThan
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            for iSubband = 2:sum(decch(2:end))
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
        end
        
        % Test for construction with order 4
        function testParameterMatrixSetRandAngMusWithDec4Ch23Ord4(testCase)
            
            % Parameters
            decch = [ 4 2 3 ];
            ord = 4;
            mstab = [ 2 2 ; 3 3 ; 2 2 ; 3 3 ; 2 2 ; 3 3 ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
            step(paramMtxExpctd,-eye(mstab(3,:)),uint32(3)); % W1
            step(paramMtxExpctd, eye(mstab(4,:)),uint32(4)); % U1
            step(paramMtxExpctd,-eye(mstab(5,:)),uint32(5)); % W2
            step(paramMtxExpctd, eye(mstab(6,:)),uint32(6)); % U2
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIIVm1System(...
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
            
            % Random angles and mus
            ang = get(testCase.lppufb,'Angles');
            ang = randn(size(ang));
            mus = get(testCase.lppufb,'Mus');
            mus = 2*(rand(size(mus))>0.5)-1;
            
            % Expected vales
            coefExpctd = 1;
            
            % Actual values
            set(testCase.lppufb,'Mus',mus);
            paramMtxActual = step(testCase.lppufb,ang,mus);
            W0 = step(paramMtxActual,[],uint32(1));
            W1 = step(paramMtxActual,[],uint32(3));
            W2 = step(paramMtxActual,[],uint32(5));
            G = W2*W1*W0;
            coefActual = G(1,1);
            
            % Evaluation
            diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
            
            % Check DC-E
            release(testCase.lppufb)
            import matlab.unittest.constraints.IsLessThan
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            for iSubband = 2:sum(decch(2:end))
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
        end

    end
    
end
