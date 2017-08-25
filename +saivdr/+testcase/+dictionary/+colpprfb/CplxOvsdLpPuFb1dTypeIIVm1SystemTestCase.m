classdef CplxOvsdLpPuFb1dTypeIIVm1SystemTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB1dTYPEIIVM1SYSTEMTESTCASE Test case for CplxOvsdLpPuFb1dTypeIIVm1System
    %
    % SVN identifier:
    % $Id: CplxOvsdLpPuFb1dTypeIIVm1SystemTestCase.m 110 2014-01-16 06:49:46Z sho $
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2015, Shogo MURAMATSU
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
                 0.353553390593274 + 0.353553390593274i, -0.353553390593274 + 0.353553390593274i,...
                -0.353553390593274 - 0.353553390593274i,  0.353553390593274 - 0.353553390593274i;
                 0.000000000000000 + 0.500000000000000i,  0.000000000000000 - 0.500000000000000i,...
                 0.000000000000000 + 0.500000000000000i,  0.000000000000000 - 0.500000000000000i;
                -0.353553390593274 + 0.353553390593274i,  0.353553390593274 + 0.353553390593274i,...
                 0.353553390593274 - 0.353553390593274i, -0.353553390593274 - 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i,  0.000000000000000 + 0.000000000000000i,...
                 0.000000000000000 + 0.000000000000000i,  0.000000000000000 + 0.000000000000000i;];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm1System();
            
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
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm1System();
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
                 0.447213595499958 + 0.000000000000000i, 0.447213595499958 + 0.000000000000000i,...
                 0.447213595499958 + 0.000000000000000i, 0.447213595499958 + 0.000000000000000i,...
                 0.447213595499958 + 0.000000000000000i;
                 0.361803398874989 + 0.262865556059567i,-0.138196601125010 + 0.425325404176020i,...
                -0.447213595499958 + 0.000000000000000i,-0.138196601125011 - 0.425325404176020i,...
                 0.361803398874989 - 0.262865556059567i;
                 0.138196601125011 + 0.425325404176020i,-0.361803398874989 - 0.262865556059567i,...
                 0.447213595499958 + 0.000000000000000i,-0.361803398874989 + 0.262865556059567i,...
                 0.138196601125010 - 0.425325404176020i;
                -0.138196601125010 + 0.425325404176020i, 0.361803398874989 - 0.262865556059567i,...
                -0.447213595499958 + 0.000000000000000i, 0.361803398874989 + 0.262865556059567i,...
                -0.138196601125011 - 0.425325404176020i;
                -0.361803398874989 + 0.262865556059567i, 0.138196601125011 + 0.425325404176020i,...
                 0.447213595499958 + 0.000000000000000i, 0.138196601125010 - 0.425325404176020i,...
                -0.361803398874989 - 0.262865556059567i];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm1System(...
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
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i,...
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 + 0.353553390593274i, -0.353553390593274 + 0.353553390593274i,...
                -0.353553390593274 - 0.353553390593274i,  0.353553390593274 - 0.353553390593274i;
                 0.000000000000000 + 0.500000000000000i,  0.000000000000000 - 0.500000000000000i,...
                 0.000000000000000 + 0.500000000000000i,  0.000000000000000 - 0.500000000000000i;
                -0.353553390593274 + 0.353553390593274i,  0.353553390593274 + 0.353553390593274i,...
                 0.353553390593274 - 0.353553390593274i, -0.353553390593274 - 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i,  0.000000000000000 + 0.000000000000000i,...
                 0.000000000000000 + 0.000000000000000i,  0.000000000000000 + 0.000000000000000i;];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm1System(...
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
            ang = 2*pi*rand(10+10,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm1System(...
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
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4
        function testConstructorWithDec4Ch32Ord4(testCase)
            
            % Parameters
            decch = [ 4 3 2 ];
            ord = 4;
            ang = [];
            
            % Expected values
            coefExpctd  = zeros(5,4,5);
            
            coefExpctd(:,:,3) = [
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i,...
                 0.500000000000000 + 0.000000000000000i,  0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 + 0.353553390593274i, -0.353553390593274 + 0.353553390593274i,...
                -0.353553390593274 - 0.353553390593274i,  0.353553390593274 - 0.353553390593274i;
                 0.000000000000000 + 0.500000000000000i,  0.000000000000000 - 0.500000000000000i,...
                 0.000000000000000 + 0.500000000000000i,  0.000000000000000 - 0.500000000000000i;
                -0.353553390593274 + 0.353553390593274i,  0.353553390593274 + 0.353553390593274i,...
                 0.353553390593274 - 0.353553390593274i, -0.353553390593274 - 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i,  0.000000000000000 + 0.000000000000000i,...
                 0.000000000000000 + 0.000000000000000i,  0.000000000000000 + 0.000000000000000i;];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm1System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end        

        % Test for construction with order 4
        function testConstructorWithDec4Ch5Ord4AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 4;
            ang = 2*pi*rand(10+2*10,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm1System(...
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

        % Test for construction with order 4
        function testConstructorWithDec4Ch5Ord8AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 8;
            ang = 2*pi*rand(10+4*10,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm1System(...
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
        function testParameterMatrixSet(testCase)
            
            % Preparation
            mstab = [ 5 5 ];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixContainer
            paramExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(5),1);
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end
        
%         % Test for construction with order 4
%         function testParameterMatrixSetRandAngWithDec4Ch32Ord4(testCase)
%             
%             % Parameters
%             decch = [ 4 3 2 ];
%             ord = 4;
%             mstab = [ 5 5 ; 2 2 ; 2 2 ; 1 1 ; 3 3 ; 3 3; 1 1 ; 2 2 ; 2 2 ; 1 1 ; 3 3 ; 3 3 ; 1 1 ];
%             
%             % Expected values
%             import saivdr.dictionary.utility.*
%             paramMtxExpctd = ParameterMatrixContainer(...
%                 'MatrixSizeTable',mstab);
%             step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
%             step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
%             step(paramMtxExpctd, eye(mstab(3,:)),uint32(3)); % W1
%             step(paramMtxExpctd,-eye(mstab(4,:)),uint32(4)); % U1
%             step(paramMtxExpctd, eye(mstab(5,:)),uint32(5)); % W2
%             step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % U2
%             coefExpctd = get(paramMtxExpctd,'Coefficients');
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm1System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord,...
%                 'OutputMode','ParameterMatrixSet');
%             
%             % Actual values
%             paramMtxActual = step(testCase.lppufb,[],[]);
%             coefActual = get(paramMtxActual,'Coefficients');
%             
%             % Evaluation
%             diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
%             
%             % Random angles
%             ang = get(testCase.lppufb,'Angles');
%             ang = randn(size(ang));
%             
%             % Expected vales
%             coefExpctd = 1;
%             
%             % Actual values
%             set(testCase.lppufb,'Angles',ang);
%             paramMtxActual = step(testCase.lppufb,ang,[]);
%             W0 = step(paramMtxActual,[],uint32(1));
%             W1 = step(paramMtxActual,[],uint32(3));
%             W2 = step(paramMtxActual,[],uint32(5));
%             G = W2*W1*W0;
%             coefActual = G(1,1);
% 
%             % Evaluation
%             diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
%             
%             % Check DC-E
%             release(testCase.lppufb)
%             import matlab.unittest.constraints.IsLessThan
%             set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
%             for iSubband = 2:sum(decch(2:end))
%                 H = step(testCase.lppufb,[],[],iSubband);
%                 dc = abs(sum(H(:)));
%                 testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
%             end
%         end

    end
    
end
