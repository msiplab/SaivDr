classdef LpPuFb2dTvmSystemTestCase < matlab.unittest.TestCase
    %LPPUFB2DTVMSYSTEMTESTCASE Test case for LpPuFb2dTvm
    %
    % SVN identifier:
    % $Id: LpPuFb2dTvmSystemTestCase.m 866 2015-11-24 04:29:42Z sho $
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
        lppufb
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.lppufb)
        end
    end
    
    methods (Test)

        % Test for default construction
        function testConstructor(testCase)
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                ];
            
            coefExpctd(:,:,2,1) = [
                -0.107390364784491   0.013640364784491   0.138640364784491   0.017609635215509
                0.415921094353473  -0.052828905646527  -0.536951823922454  -0.068201823922454
                0.000000000000000  -0.000000000000000                   0                   0
                0.429561459137964  -0.054561459137964  -0.554561459137964  -0.070438540862036
                ];
            
            coefExpctd(:,:,3,1) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                ];
            
            coefExpctd(:,:,1,2) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                ];
            
            coefExpctd(:,:,2,2) = [
                0.589780729568982   0.347719270431018   0.347719270431018   0.589780729568982
                0.152280729568982   0.089780729568982   0.089780729568982   0.152280729568982
                0.500000000000000  -0.500000000000000   0.500000000000000  -0.500000000000000
                0   0.000000000000000                   0                   0
                ];
            
            coefExpctd(:,:,3,2) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                ];
            
            coefExpctd(:,:,1,3) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                ];
            
            coefExpctd(:,:,2,3) = [
                0.017609635215509   0.138640364784491   0.013640364784491  -0.107390364784491
                -0.068201823922454  -0.536951823922454  -0.052828905646527   0.415921094353473
                0   0.000000000000000   0.000000000000000  -0.000000000000000
                0.070438540862036   0.554561459137964   0.054561459137964  -0.429561459137964
                ];
            
            coefExpctd(:,:,3,3) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            testCase.lppufb = LpPuFb2dTvmSystem();
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-15,...
                sprintf('%g',coefDist));
            
        end

        % Test for default construction
        function testConstructorWithDeepCopy(testCase)
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            testCase.lppufb = LpPuFb2dTvmSystem();
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
        function testConstructorWithOrd00(testCase)
            
            % Parameters
            phi = 0;
            dec = [ 2 2 ];
            ord = [ 0 0 ];
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Order must be greater than or equal to 2');
            
            % Instantiation of target class
            try
                import saivdr.dictionary.nsgenlotx.*
                testCase.lppufb = LpPuFb2dTvmSystem(...
                    'DecimationFactor',dec,...
                    'PolyPhaseOrder',ord,...
                    'TvmAngleInDegree',phi);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end            
        end

        % Test for construction with order 2 2 and directional vm
        function testConstructorWithDec44Ord22Tvm120(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(28,6);
            mus = 2*round(rand(8,6))-1;
            phi = 120;
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % lenx3
            lenx3 = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(coefActual,dimExpctd);
            testCase.verifyThat(lenx3,IsLessThan(2));
            
            % Check symmetry
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
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check DC-leakage
            release(testCase.lppufb)
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            H = step(testCase.lppufb,[],[],1);
            dc = abs(sum(H(:)));
            testCase.verifyEqual(dc,sqrt(prod(dec)),'RelTol',1e-14,...
                sprintf('%g',dc));
            for iSubband = 2:prod(dec)
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
            % Check Directional VM
             if get(testCase.lppufb,'lambdaueq') <= 0
                 [ckTvm, angleVm] = checkTvm(testCase.lppufb);
                 testCase.verifyThat(angleVm-phi,IsLessThan(1e-15),...
                     sprintf('a = %g: invalid angle (neq %g)', angleVm, phi));
                 testCase.verifyThat(ckTvm,IsLessThan(1e-15),...
                     sprintf('c = %g: triangle condition failed', ckTvm));
             end
        end
                
        function testConstructorWithInvalidArguments(testCase)
            
            % Invalid input
            phi = 0;
            dec = [ 4 4 ];
            ord = [ 2 2 ];
            sizeInvalid = [2 2];
            ang = 2*pi*rand(sizeInvalid);
            
            % Expected value
            sizeExpctd = [28 6];
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Size of angles must be [ %d %d ]',...
                sizeExpctd(1), sizeExpctd(2));
            
            % Instantiation of target class
            try
                import saivdr.dictionary.nsgenlotx.*
                testCase.lppufb = LpPuFb2dTvmSystem(...
                    'DecimationFactor',dec,...
                    'PolyPhaseOrder',ord,...
                    'TvmAngleInDegree',phi);
                step(testCase.lppufb,ang,[]);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual,messageExpctd);
            end
        end
        
        %{
        % Test for char
        function testChar(testCase)
            
            % Expected value
            if isunix || ~verLessThan('matlab', '7.9')
                charExpctd = [...
                    '[',10,...
                    9,'-0.10739*y^(-1) + 0.01364*y^(-2) + 0.72842*y^(-1)*x^(-1) + 0.36533*y^(-2)*x^(-1) + 0.36533*y^(-1)*x^(-2) + 0.72842*y^(-2)*x^(-2) + 0.01364*y^(-1)*x^(-3) - 0.10739*y^(-2)*x^(-3);',10,...
                    9,'0.41592*y^(-1) - 0.052829*y^(-2) - 0.38467*y^(-1)*x^(-1) + 0.021579*y^(-2)*x^(-1) + 0.021579*y^(-1)*x^(-2) - 0.38467*y^(-2)*x^(-2) - 0.052829*y^(-1)*x^(-3) + 0.41592*y^(-2)*x^(-3);',10,...
                    9,'2.7756e-17*y^(-1) - 3.4694e-18*y^(-2) + 0.5*y^(-1)*x^(-1) - 0.5*y^(-2)*x^(-1) + 0.5*y^(-1)*x^(-2) - 0.5*y^(-2)*x^(-2) + 1.7347e-18*y^(-1)*x^(-3) - 2.7756e-17*y^(-2)*x^(-3);',10,...
                    9,'0.42956*y^(-1) - 0.054561*y^(-2) - 0.55456*y^(-1)*x^(-1) - 0.070439*y^(-2)*x^(-1) + 0.070439*y^(-1)*x^(-2) + 0.55456*y^(-2)*x^(-2) + 0.054561*y^(-1)*x^(-3) - 0.42956*y^(-2)*x^(-3)',10,...
                    ']'...
                    ];
            else
                charExpctd = [...
                    '[',10,...
                    9,'-0.10739*y^(-1) + 0.01364*y^(-2) + 0.72842*y^(-1)*x^(-1) + 0.36533*y^(-2)*x^(-1) + 0.36533*y^(-1)*x^(-2) + 0.72842*y^(-2)*x^(-2) + 0.01364*y^(-1)*x^(-3) - 0.10739*y^(-2)*x^(-3);',10,...
                    9,'0.41592*y^(-1) - 0.052829*y^(-2) - 0.38467*y^(-1)*x^(-1) + 0.021579*y^(-2)*x^(-1) + 0.021579*y^(-1)*x^(-2) - 0.38467*y^(-2)*x^(-2) - 0.052829*y^(-1)*x^(-3) + 0.41592*y^(-2)*x^(-3);',10,...
                    9,'2.7756e-017*y^(-1) - 3.4694e-018*y^(-2) + 0.5*y^(-1)*x^(-1) - 0.5*y^(-2)*x^(-1) + 0.5*y^(-1)*x^(-2) - 0.5*y^(-2)*x^(-2) + 1.7347e-018*y^(-1)*x^(-3) - 2.7756e-017*y^(-2)*x^(-3);',10,...
                    9,'0.42956*y^(-1) - 0.054561*y^(-2) - 0.55456*y^(-1)*x^(-1) - 0.070439*y^(-2)*x^(-1) + 0.070439*y^(-1)*x^(-2) + 0.55456*y^(-2)*x^(-2) + 0.054561*y^(-1)*x^(-3) - 0.42956*y^(-2)*x^(-3)',10,...
                    ']'...
                    ];
            end
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'OutputMode','Char');
            
            % Actual values
            charActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(charActual, charExpctd);
            
        end
        %}
        
        % Test for construction with order 2 2 and directional vm
        function testConstructorWithDec22Ord22Tvm0(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 2 2 ];
            ang = (2*pi*rand(1,6)-1);
            mus = 2*round(rand(2,6))-1;
            phi = 0;
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
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
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check DC-leakage
            release(testCase.lppufb)
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            H = step(testCase.lppufb,[],[],1);
            dc = abs(sum(H(:)));
            testCase.verifyEqual(dc,sqrt(prod(dec)),'RelTol',1e-15,...
                sprintf('%g',dc));
            for iSubband = 2:prod(dec)
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
            % Check Directional VM
            [ckTvm, angleVm] = checkTvm(testCase.lppufb);
            testCase.verifyThat(angleVm,IsLessThan(1e-15),...
                sprintf('a = %g: invalid angle', angleVm));
            testCase.verifyThat(ckTvm,IsLessThan(1e-15),...
                sprintf('c = %g: triangle condition failed', ckTvm));
            
        end

        % Test for construction with order 2 2 and directional vm
        function testConstructorWithDec44Ord22Tvm0(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(28,6);
            mus = 2*round(rand(8,6))-1;
            phi = 0;
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
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
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check DC-leakage
            release(testCase.lppufb)
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            H = step(testCase.lppufb,[],[],1);
            dc = abs(sum(H(:)));
            testCase.verifyEqual(dc,sqrt(prod(dec)),'RelTol',1e-14,...
                sprintf('%g',dc));
            for iSubband = 2:prod(dec)
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
            % Check Directional VM
            if get(testCase.lppufb,'lambdaueq') <= 0
                [ckTvm, angleVm] = checkTvm(testCase.lppufb);
                testCase.verifyThat(angleVm,IsLessThan(1e-15),...
                    sprintf('a = %g: invalid angle', angleVm));
                testCase.verifyThat(ckTvm,IsLessThan(1e-15),...
                    sprintf('c = %g: triangle condition failed', ckTvm));
            end
        end

        % Test for construction with order 3 3 and directional vm
        function testConstructorWithDec22Ord33Tvm0(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 3 3 ];
            ang = (2*pi*rand(1,8)-1);
            mus = 2*round(rand(2,8))-1;
            phi = 0;
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % lenx3
            lenx3 = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(coefActual,dimExpctd);
            testCase.verifyThat(lenx3,IsLessThan(2));
            
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
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check DC-leakage
            release(testCase.lppufb)
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            H = step(testCase.lppufb,[],[],1);
            dc = abs(sum(H(:)));
            testCase.verifyEqual(dc,sqrt(prod(dec)),'RelTol',1e-15,...
                sprintf('%g',dc));
            for iSubband = 2:prod(dec)
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
            % Check Directional VM
            [ckTvm, angleVm] = checkTvm(testCase.lppufb);
            testCase.verifyThat(angleVm,IsLessThan(1e-15),...
                sprintf('a = %g: invalid angle', angleVm));
            testCase.verifyThat(ckTvm,IsLessThan(1e-15),...
                sprintf('c = %g: triangle condition failed', ckTvm));
            
        end
        
        % Test for construction with order 2 2 and directional vm
        function testConstructorWithDec44Ord33Tvm0(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 3 3 ];
            ang = 2*pi*rand(28,8);
            mus = 2*round(rand(8,8))-1;
            phi = 0;
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % lenx3
            lenx3 = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(coefActual,dimExpctd);
            testCase.verifyThat(lenx3,IsLessThan(2));
            
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
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check DC-leakage
            release(testCase.lppufb)
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            H = step(testCase.lppufb,[],[],1);
            dc = abs(sum(H(:)));
            testCase.verifyEqual(dc,sqrt(prod(dec)),'RelTol',1e-14,...
                sprintf('%g',dc));
            for iSubband = 2:prod(dec)
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
            % Check Directional VM
            if get(testCase.lppufb,'lambdaueq') <= 0
                [ckTvm, angleVm] = checkTvm(testCase.lppufb);
                testCase.verifyThat(angleVm,IsLessThan(1e-15),...
                    sprintf('a = %g: invalid angle', angleVm));
                testCase.verifyThat(ckTvm,IsLessThan(1e-15),...
                    sprintf('c = %g: triangle condition failed', ckTvm));
            end
            
        end

        % Test for construction with order 2 2 and directional vm
        function testConstructorWithDec22Ord22Tvm60(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 2 2 ];
            ang = (2*pi*rand(1,6)-1);
            mus = 2*round(rand(2,6))-1;
            phi = 60;
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % lenx3
            lenx3 = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(coefActual,dimExpctd);
            testCase.verifyThat(lenx3,IsLessThan(2));
            
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
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check DC-leakage
            release(testCase.lppufb)
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            H = step(testCase.lppufb,[],[],1);
            dc = abs(sum(H(:)));
            testCase.verifyEqual(dc,sqrt(prod(dec)),'RelTol',1e-15,...
                sprintf('%g',dc));
            for iSubband = 2:prod(dec)
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
            % Check Directional VM
            [ckTvm, angleVm] = checkTvm(testCase.lppufb);
            testCase.verifyThat(angleVm-phi,IsLessThan(1e-15),...
                sprintf('a = %g: invalid angle (neq %g)', angleVm, phi));
            testCase.verifyThat(ckTvm,IsLessThan(1e-15),...
                sprintf('c = %g: triangle condition failed', ckTvm));
            
        end

        % Test for construction with order 2 2 and directional vm
        function testConstructorWithDeepCopyDec44Ord22Tvm120(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(28,6);
            mus = 2*round(rand(8,6))-1;
            phi = 120;
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            cloneLpPuFb = clone(testCase.lppufb);
            
            % Evaluation
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.PublicPropertyComparator
            %testCase.verifyEqual(cloneLpPuFb,testCase.lppufb);
            testCase.verifyThat(cloneLpPuFb, IsEqualTo(testCase.lppufb,...
            'Using', PublicPropertyComparator.supportingAllValues))            
            testCase.verifyFalse(cloneLpPuFb == testCase.lppufb);
            prpOrg = get(testCase.lppufb,'ParameterMatrixSet');
            prpCln = get(cloneLpPuFb,'ParameterMatrixSet');
            %testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyThat(prpCln, IsEqualTo(prpOrg,...
            'Using', PublicPropertyComparator.supportingAllValues))                        
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
        
        % Test for construction with order 3 3 and directional vm
        function testConstructorWithDec22Ord33Tvm90(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 3 3 ];
            ang = (2*pi*rand(1,8)-1);
            mus = 2*round(rand(2,8))-1;
            phi = 90;
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % lenx3
            lenx3 = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(coefActual,dimExpctd);
            testCase.verifyThat(lenx3,IsLessThan(2));
            
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
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check DC-leakage
            release(testCase.lppufb)
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            H = step(testCase.lppufb,[],[],1);
            dc = abs(sum(H(:)));
            testCase.verifyEqual(dc,sqrt(prod(dec)),'RelTol',1e-14,...
                sprintf('%g',dc));
            for iSubband = 2:prod(dec)
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
            % Check Directional VM
            [ckTvm, angleVm] = checkTvm(testCase.lppufb);
            testCase.verifyThat(angleVm-phi,IsLessThan(1e-15),...
                sprintf('a = %g: invalid angle (neq %g)', angleVm, phi));
            testCase.verifyThat(ckTvm,IsLessThan(1e-15),...
                sprintf('c = %g: triangle condition failed', ckTvm));
                        
        end

        % Test for construction with order 3 3 and directional vm
        function testConstructorWithDec44Ord33Tvm90(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 3 3 ];
            ang = 2*pi*rand(28,8);
            mus = 2*round(rand(8,8))-1;
            phi = 90;
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % lenx3
            lenx3 = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(coefActual,dimExpctd);
            testCase.verifyThat(lenx3,IsLessThan(2));
            
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
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),...
                sprintf('%g',coefDist));
            
            % Check DC-leakage
            release(testCase.lppufb)
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            H = step(testCase.lppufb,[],[],1);
            dc = abs(sum(H(:)));
            testCase.verifyEqual(dc,sqrt(prod(dec)),'RelTol',1e-12,...
                sprintf('%g',dc));
            for iSubband = 2:prod(dec)
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
            % Check Directional VM
            if get(testCase.lppufb,'lambdaueq') <= 0
                [ckTvm, angleVm] = checkTvm(testCase.lppufb);
                testCase.verifyThat(angleVm-phi,IsLessThan(1e-15),...
                    sprintf('a = %g: invalid angle (neq %g)', angleVm, phi));
                testCase.verifyThat(ckTvm,IsLessThan(1e-15),...
                    sprintf('c = %g: triangle condition failed', ckTvm));
            end
            
        end

        % Test for construction with order 2 2 and directional vm
        function testGetLengthX3Dec22Ord22Tvm(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 2 2 ];
            ang = zeros(1,6);
            mus = ones(2,6);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            phi = 0;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 0.5;
            lambdaUeqExpctd = -0.75;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 30;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 1;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            
            % Instantiation of target class
            phi = 60;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 0.715518083829763;
            lambdaUeqExpctd = -0.642240958085119;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 90;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 0.5;
            lambdaUeqExpctd = -0.75;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 120;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 1.679692592424553;
            lambdaUeqExpctd = -0.160153703787724;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 150;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 1;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
        end

        % Test for construction with order 3 3 and directional vm
        function testGetLengthX3Dec22Ord33Tvm(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 3 3 ];
            ang = zeros(1,8);
            mus = ones(2,8);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            phi = 0;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 1.118033988749895;
            lambdaUeqExpctd = -0.440983005625053;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 30;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd =   2.494009759259467; % > 2 (violated)
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-14,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            
            % Instantiation of target class
            phi = 60;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 2.250640828942328; % > 2 (violated)
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-14,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            
            % Instantiation of target class
            phi = 90;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 0.5;
            lambdaUeqExpctd = -0.75;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 120;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 1.265417925337101;
            lambdaUeqExpctd = -0.367291037331449;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 150;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd =  0.668267900908913;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            
        end

        % Test for construction with order 2 2 and directional vm
        function testGetLengthX3Dec44Ord22Tvm(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 2 2 ];
            ang = zeros(28,6);
            mus = ones(8,6);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            phi = 0;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 0.559016994374947;
            lambdaUeqExpctd = -0.720491502812526;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 30;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 1.003254288571669; %
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            
            % Instantiation of target class
            phi = 60;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 0.679892214782058;
            lambdaUeqExpctd = -0.660053892608971;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 90;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 0.559016994374947;
            lambdaUeqExpctd = -0.720491502812526;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 120;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 1.742913244048236;
            lambdaUeqExpctd = -0.128543377975882;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 150;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 1.003254288571669;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
        end

        % Test for construction with order 3 3 and directional vm
        function testGetLengthX3Dec44Ord33Tvm(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 3 3 ];
            ang = zeros(28,8);
            mus = ones(8,8);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.*
            phi = 0;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 1.145643923738960;
            lambdaUeqExpctd = -0.427178038130520;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 30;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 2.474201637896802; % > 2 (violated)
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-14,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            
            % Instantiation of target class
            phi = 60;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 2.198619377857948; % > 2 (violated)
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            
            % Instantiation of target class
            phi = 90;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 0.444160726884051;
            lambdaUeqExpctd = -0.777919636557975;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 120;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 1.330023934842227;
            lambdaUeqExpctd = -0.334988032578886;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            lambdaUeqActual = get(testCase.lppufb,'lambdaueq');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            testCase.verifyEqual(lambdaUeqActual,lambdaUeqExpctd,...
                'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid projection of x3 (neq %g)', ...
                phi, lambdaUeqActual,lambdaUeqExpctd));
            
            % Instantiation of target class
            phi = 150;
            testCase.lppufb = LpPuFb2dTvmSystem(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'TvmAngleInDegree',phi);
            step(testCase.lppufb,ang,mus);
            
            % Expected values
            lenx3Expctd = 0.693698126690893;
            
            % Actual values
            lenx3Actual = get(testCase.lppufb,'lenx3');
            
            % Evaluation
            testCase.verifyEqual(lenx3Actual,lenx3Expctd,'AbsTol',1e-15,...
                sprintf('(%g) %g: invalid length of x3 (neq %g)', ...
                phi, lenx3Actual,lenx3Expctd));
            
        end

        function testInvalidParameters(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Unsupported combination of PHI and ORD');
            
            % Instantiation of target class
            try
                import saivdr.dictionary.nsgenlotx.*
                ord = [2 0];
                phi = 0;
                testCase.lppufb = LpPuFb2dTvmSystem(...
                    'DecimationFactor',dec,....
                    'PolyPhaseOrder',ord,...
                    'TvmAngleInDegree',phi);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
            
            % Instantiation of target class
            try
                import saivdr.dictionary.nsgenlotx.*
                ord = [1 1];
                phi = 45;
                testCase.lppufb = LpPuFb2dTvmSystem(...
                    'DecimationFactor',dec,....
                    'PolyPhaseOrder',ord,...
                    'TvmAngleInDegree',phi);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
            
            % Instantiation of target class
            try
                import saivdr.dictionary.nsgenlotx.*                
                ord = [0 2];
                phi = 90;
                testCase.lppufb = LpPuFb2dTvmSystem(...
                    'DecimationFactor',dec,....
                    'PolyPhaseOrder',ord,...
                    'TvmAngleInDegree',phi);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
        end

        function testTrendSurfaceAnnihilation(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            dim = [ 64 64 ];

            import saivdr.dictionary.nsgenlotx.*
            import matlab.unittest.constraints.IsLessThan
            for phi = 0:5:180
                % Instantiation of target class
                testCase.lppufb = LpPuFb2dTvmSystem(...
                    'DecimationFactor',dec,...
                    'TvmAngleInDegree',phi,...
                    'OutputMode','AnalysisFilterAt');
                
                % Trend Surface
                I = NsGenLotUtility.trendSurface(phi,dim);
                
                for iSubband = 2:prod(dec)
                    J = conv2(I,step(testCase.lppufb,[],[],iSubband));
                    subJ = J(dim(1)/4:dim(1)*3/4,dim(2)/4:dim(2)*3/4);
                    mse = norm(subJ(:))/numel(subJ);
                    msg = ...
                        sprintf('phi = %g, sb = %d, mse = %g, lenx3 = %g', ...
                        phi, iSubband, mse, get(testCase.lppufb,'lenx3'));
                    %fprintf('%s\n',msg);
                    % Evaluation
                    testCase.verifyThat(mse,IsLessThan(1e-15),msg);
                end
            end
            
        end
        
        function testTrendSurfaceAnnihilationOrd2(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            dim = [ 64 64 ];

            import saivdr.dictionary.nsgenlotx.*
            import matlab.unittest.constraints.IsLessThan
            for phi = 0:5:180
                
                % Instantiation of target class
                if mod(phi+45,180)-45 < 45
                    ord = [ 0 2 ];
                else
                    ord = [ 2 0 ];
                end
                testCase.lppufb = LpPuFb2dTvmSystem(....
                    'DecimationFactor',dec,...
                    'PolyPhaseOrder',ord,...
                    'TvmAngleInDegree',phi,...
                    'OutputMode','AnalysisFilterAt');
                
                % Trend Surface
                I = NsGenLotUtility.trendSurface(phi,dim);
                
                for iSubband = 2:prod(dec)
                    J = conv2(I,step(testCase.lppufb,[],[],iSubband));
                    subJ = J(dim(1)/4:dim(1)*3/4,dim(2)/4:dim(2)*3/4);
                    mse = norm(subJ(:))/numel(subJ);
                    msg = ...
                        sprintf('phi = %g, sb = %d, mse = %g, lenx3 = %g', ...
                        phi, iSubband, mse, get(testCase.lppufb,'lenx3'));
                    %fprintf('%s\n',msg);
                    % Evaluation
                    testCase.verifyThat(mse,IsLessThan(1e-15),msg);
                end
            end
            
        end
        
        function testTrendSurfaceAnnihilationDec44(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            dim = [ 64 64 ];
            
            import saivdr.dictionary.nsgenlotx.*
            import matlab.unittest.constraints.IsLessThan
            for phi = 0:5:180
                % Instantiation of target class
                testCase.lppufb = LpPuFb2dTvmSystem(...
                    'DecimationFactor',dec,...
                    'TvmAngleInDegree',phi,...
                    'OutputMode','AnalysisFilterAt');
                
                % Trend Surface
                I = NsGenLotUtility.trendSurface(phi,dim);
                
                for iSubband = 2:prod(dec)
                    J = conv2(I,step(testCase.lppufb,[],[],iSubband));
                    subJ = J(dim(1)/4:dim(1)*3/4,dim(2)/4:dim(2)*3/4);
                    mse = norm(subJ(:))/numel(subJ);
                    msg = ...
                        sprintf('phi = %g, sb = %d, mse = %g, lenx3 = %g', ...
                        phi, iSubband, mse, get(testCase.lppufb,'lenx3'));
                    %fprintf('%s\n',msg);
                    % Evaluation
                    testCase.verifyThat(mse,IsLessThan(1e-15),msg);
                end
            end
            
        end
        
        function testTrendSurfaceAnnihilationOrd4(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ang = 2*pi*rand(1,6);
            mus = 2*round(rand(2,6))-1;
            sl = 2*round(rand(1))-1;
            dim = [ 64 64 ];
            
            import saivdr.dictionary.nsgenlotx.*
            import matlab.unittest.constraints.IsLessThan
            for phi = 0:5:180
                % Instantiation of target class
                if mod(phi+45,180)-45 < 45
                    ord = [ 0 4 ];
                else
                    ord = [ 4 0 ];
                end
                testCase.lppufb = LpPuFb2dTvmSystem(...
                    'DecimationFactor',dec,...
                    'PolyPhaseOrder',ord,...
                    'TvmAngleInDegree',phi,...
                    'DirectionOfTriangle',sl,...
                    'OutputMode','AnalysisFilterAt');
                step(testCase.lppufb,ang,mus,1);
                
                % Trend Surface
                I = NsGenLotUtility.trendSurface(phi,dim);

                if (get(testCase.lppufb,'lenx3') < 2) && ...
                        (get(testCase.lppufb,'lambdaueq') <= 0)
                    for iSubband = 2:prod(dec)
                        J = conv2(I,step(testCase.lppufb,ang,mus,iSubband));
                        subJ = J(dim(1)/4:dim(1)*3/4,dim(2)/4:dim(2)*3/4);
                        mse = norm(subJ(:))/numel(subJ);
                        msg = ...
                            sprintf('phi = %g, sb = %d, mse = %g, lenx3 = %g, lambdaueq = %g', ...
                            phi, iSubband, mse, ...
                            get(testCase.lppufb,'lenx3'),...
                            get(testCase.lppufb,'lambdaueq'));
                        
                        %fprintf('%s\n',msg);
                        % Evaluation
                        testCase.verifyThat(mse,IsLessThan(1e-15),msg);
                    end
                end
            end
        end
        
        function testTrendSurfaceAnnihilationOrd44(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(1,10);
            mus = 2*round(rand(2,10))-1;
            sl = 2*round(rand(1))-1;
            dim = [ 64 64 ];
            
            import saivdr.dictionary.nsgenlotx.*
            import matlab.unittest.constraints.IsLessThan            
            for phi = 0:5:180
                % Instantiation of target class
                testCase.lppufb = LpPuFb2dTvmSystem(...
                    'DecimationFactor',dec,...
                    'PolyPhaseOrder',ord,...
                    'TvmAngleInDegree',phi,...
                    'DirectionOfTriangle',sl,...
                    'OutputMode','AnalysisFilterAt');
                step(testCase.lppufb,ang,mus,1);
                
                % Trend Surface
                I = NsGenLotUtility.trendSurface(phi,dim);
                
                if get(testCase.lppufb,'lenx3') < 2 && ...
                        get(testCase.lppufb,'lambdaueq') <= 0
                    for iSubband = 2:prod(dec)
                        J = conv2(I,step(testCase.lppufb,ang,mus,iSubband));
                        subJ = J(dim(1)/4:dim(1)*3/4,dim(2)/4:dim(2)*3/4);
                        mse = norm(subJ(:))/numel(subJ);
                        msg = ...
                            sprintf('phi = %g, sb = %d, mse = %g, lenx3 = %g, lambdaueq = %g', ...
                            phi, iSubband, mse, ...
                            get(testCase.lppufb,'lenx3'),...
                            get(testCase.lppufb,'lambdaueq'));
                        
                        %fprintf('%s\n',msg);
                        % Evaluation
                        testCase.verifyThat(mse,IsLessThan(1e-15),msg);
                    end
                end
            end
        end
        
        function testTrendSurfaceAnnihilationDec44Ord44(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 4 4 ];
            ang = 0; %2*pi*rand(28,10);
            mus = 2*round(rand(8,10))-1;
            sl = 2*round(rand(1))-1;
            dim = [ 128 128 ];
        
            import saivdr.dictionary.nsgenlotx.*
            import matlab.unittest.constraints.IsLessThan                        
            for phi = 0:5:180
                % Instantiation of target class
                testCase.lppufb = LpPuFb2dTvmSystem(...
                    'DecimationFactor',dec,...
                    'PolyPhaseOrder',ord,...
                    'TvmAngleInDegree',phi,...
                    'DirectionOfTriangle',sl,...
                    'OutputMode','AnalysisFilterAt');
                step(testCase.lppufb,ang,mus,1);
                
                % Trend Surface
                I = NsGenLotUtility.trendSurface(phi,dim);
                
                if get(testCase.lppufb,'lenx3') < 2 && ...
                        get(testCase.lppufb,'lambdaueq') <= 0
                    for iSubband = 2:prod(dec)
                        J = conv2(I,step(testCase.lppufb,ang,mus,iSubband));
                        subJ = J(dim(1)/4:dim(1)*3/4,dim(2)/4:dim(2)*3/4);
                        mse = norm(subJ(:))/numel(subJ);
                        msg = ...
                            sprintf('phi = %g, sb = %d, mse = %g, lenx3 = %g, lambdaueq = %g', ...
                            phi, iSubband, mse, ...
                            get(testCase.lppufb,'lenx3'),...
                            get(testCase.lppufb,'lambdaueq'));
                        
                        %fprintf('%s\n',msg);
                        % Evaluation
                        testCase.verifyThat(mse,IsLessThan(1e-15),msg);
                    end
                end
            end
            
        end

    end
end
