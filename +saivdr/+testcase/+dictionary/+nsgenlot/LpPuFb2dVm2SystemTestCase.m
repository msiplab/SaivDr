classdef LpPuFb2dVm2SystemTestCase < matlab.unittest.TestCase
    %LPPUFB2DVM2SYSTEMTESTCASE Test case for LpPuFb2dVm2
    %
    % SVN identifier:
    % $Id: LpPuFb2dVm2SystemTestCase.m 683 2015-05-29 08:22:13Z sho $
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
                0.006820182392245  -0.053695182392245   0.008804817607755   0.069320182392245
                0.026414452823264  -0.207960547176736   0.034100911961227   0.268475911961227
                -0.006820182392245   0.053695182392245  -0.008804817607755  -0.069320182392245
                -0.026414452823264   0.207960547176736  -0.034100911961227  -0.268475911961227
                ];
            
            coefExpctd(:,:,2,1) = [
                0.000000000000000                   0  -0.000000000000000                   0
                -0.000000000000000  -0.000000000000000                   0  -0.000000000000000
                -0.052828905646527   0.415921094353473  -0.068201823922454  -0.536951823922455
                0.013640364784491  -0.107390364784491   0.017609635215509   0.138640364784491
                ];
            
            coefExpctd(:,:,3,1) = [
                0.006820182392245  -0.053695182392245   0.008804817607755   0.069320182392245
                0.026414452823264  -0.207960547176736   0.034100911961227   0.268475911961227
                0.006820182392245  -0.053695182392245   0.008804817607755   0.069320182392245
                0.026414452823264  -0.207960547176736   0.034100911961227   0.268475911961227
                ];
            
            coefExpctd(:,:,1,2) = [
                -0.062500000000000   0.062500000000000  -0.062500000000000   0.062500000000000
                -0.242061459137964   0.242061459137964  -0.242061459137964   0.242061459137964
                0.062500000000000  -0.062500000000000   0.062500000000000  -0.062500000000000
                0.242061459137964  -0.242061459137964   0.242061459137964  -0.242061459137964
                ];
            
            coefExpctd(:,:,2,2) = [
                0.347719270431018   0.589780729568982   0.589780729568982   0.347719270431018
                -0.089780729568982  -0.152280729568982  -0.152280729568982  -0.089780729568982
                0                   0  -0.000000000000000   0.000000000000000
                -0.000000000000000                   0   0.000000000000000  -0.000000000000000
                ];
            
            coefExpctd(:,:,3,2) = [
                0.062500000000000  -0.062500000000000   0.062500000000000  -0.062500000000000
                0.242061459137964  -0.242061459137964   0.242061459137964  -0.242061459137964
                0.062500000000000  -0.062500000000000   0.062500000000000  -0.062500000000000
                0.242061459137964  -0.242061459137964   0.242061459137964  -0.242061459137964
                ];
            
            coefExpctd(:,:,1,3) = [
                0.069320182392245   0.008804817607755  -0.053695182392245   0.006820182392245
                0.268475911961227   0.034100911961227  -0.207960547176736   0.026414452823264
                -0.069320182392245  -0.008804817607755   0.053695182392245  -0.006820182392245
                -0.268475911961227  -0.034100911961227   0.207960547176736  -0.026414452823264
                ];
            
            coefExpctd(:,:,2,3) = [
                0.000000000000000                   0   0.000000000000000                   0
                -0.000000000000000  -0.000000000000000                   0                   0
                0.536951823922455   0.068201823922454  -0.415921094353473   0.052828905646527
                -0.138640364784491  -0.017609635215509   0.107390364784491  -0.013640364784491
                ];
            
            coefExpctd(:,:,3,3) = [
                0.069320182392245   0.008804817607755  -0.053695182392245   0.006820182392245
                0.268475911961227   0.034100911961227  -0.207960547176736   0.026414452823264
                0.069320182392245   0.008804817607755  -0.053695182392245   0.006820182392245
                0.268475911961227   0.034100911961227  -0.207960547176736   0.026414452823264
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*
            testCase.lppufb = LpPuFb2dVm2System();
            
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
            import saivdr.dictionary.nsgenlot.*
            testCase.lppufb = LpPuFb2dVm2System();
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
            
            % Invalid input
            dec = [ 2 2 ];
            ord = [ 0 0 ];
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Order must be greater than or equal to [ 2 2 ]');
            
            % Instantiation of target class
            try
                import saivdr.dictionary.nsgenlot.*
                testCase.lppufb = LpPuFb2dVm2System(...
                    'DecimationFactor',dec,...
                    'PolyPhaseOrder',ord);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
        end
        
        function testConstructorWithInvalidArguments(testCase)
            
            % Invalid input
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
                import saivdr.dictionary.nsgenlot.*
                testCase.lppufb = LpPuFb2dVm2System(...
                    'DecimationFactor',dec,...
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
        
        function testChar(testCase)
            
            % Expected value
            charExpctd = [...
                '[', 10, ...
                9, '0.0068202 - 0.053695*y^(-1) + 0.0068202*y^(-2) - 0.053695*y^(-3) - 0.053695*x^(-1) + 0.47954*y^(-1)*x^(-1) + 0.66109*y^(-2)*x^(-1) + 0.0068202*y^(-3)*x^(-1) + 0.0068202*x^(-2) + 0.66109*y^(-1)*x^(-2) + 0.47954*y^(-2)*x^(-2) - 0.053695*y^(-3)*x^(-2) - 0.053695*x^(-3) + 0.0068202*y^(-1)*x^(-3) - 0.053695*y^(-2)*x^(-3) + 0.0068202*y^(-3)*x^(-3);',10,...
                9, '0.026414 - 0.20796*y^(-1) + 0.026414*y^(-2) - 0.20796*y^(-3) - 0.20796*x^(-1) + 0.42076*y^(-1)*x^(-1) + 0.12388*y^(-2)*x^(-1) + 0.026414*y^(-3)*x^(-1) + 0.026414*x^(-2) + 0.12388*y^(-1)*x^(-2) + 0.42076*y^(-2)*x^(-2) - 0.20796*y^(-3)*x^(-2) - 0.20796*x^(-3) + 0.026414*y^(-1)*x^(-3) - 0.20796*y^(-2)*x^(-3) + 0.026414*y^(-3)*x^(-3);', 10, ...
                9, '-0.0068202 + 0.00086628*y^(-1) + 0.42274*y^(-2) - 0.053695*y^(-3) + 0.053695*x^(-1) - 0.20002*y^(-1)*x^(-1) - 0.46565*y^(-2)*x^(-1) + 0.0068202*y^(-3)*x^(-1) - 0.0068202*x^(-2) + 0.46565*y^(-1)*x^(-2) + 0.20002*y^(-2)*x^(-2) - 0.053695*y^(-3)*x^(-2) + 0.053695*x^(-3) - 0.42274*y^(-1)*x^(-3) - 0.00086628*y^(-2)*x^(-3) + 0.0068202*y^(-3)*x^(-3);',10,...
                9, '-0.026414 + 0.2216*y^(-1) - 0.080976*y^(-2) - 0.20796*y^(-3) + 0.20796*x^(-1) - 0.49293*y^(-1)*x^(-1) + 0.4148*y^(-2)*x^(-1) + 0.026414*y^(-3)*x^(-1) - 0.026414*x^(-2) - 0.4148*y^(-1)*x^(-2) + 0.49293*y^(-2)*x^(-2) - 0.20796*y^(-3)*x^(-2) + 0.20796*x^(-3) + 0.080976*y^(-1)*x^(-3) - 0.2216*y^(-2)*x^(-3) + 0.026414*y^(-3)*x^(-3)',10,...
                ']' ...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*
            testCase.lppufb = LpPuFb2dVm2System(...
                'OutputMode','Char');
            
            % Actual values
            charActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(charActual, charExpctd);
            
        end

        % Test for construction with order 2 2
        function testConstructorWithDec22Ord22Vm2(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 2 2 ];
            ang = (2*pi*rand(1,6)-1);
            mus = 2*round(rand(2,6))-1;
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*
            testCase.lppufb = LpPuFb2dVm2System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
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
            for iSubband = 2:prod(dec)
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end

            % Check Vm2
            ckVm2 = checkVm2(testCase.lppufb);
            testCase.verifyThat(ckVm2(1),IsLessThan(1e-15),...
                sprintf('cy=%g: triange condition failed',ckVm2(1)));
            testCase.verifyThat(ckVm2(2),IsLessThan(1e-15),...
                sprintf('cx=%g: triange condition failed',ckVm2(2)));
        end

        % Test for construction with order 2 2
        function testConstructorWithDec44Ord22Vm2(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(28,6);
            mus = 2*round(rand(8,6))-1;
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*
            testCase.lppufb = LpPuFb2dVm2System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
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
            for iSubband = 2:prod(dec)
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end

            % Check Vm2
            if get(testCase.lppufb,'lambdaxueq') <=0 && ...
                    get(testCase.lppufb,'lambdayueq') <=0
                ckVm2 = checkVm2(testCase.lppufb);
                testCase.verifyThat(ckVm2(1),IsLessThan(1e-15),...
                    sprintf('cy=%g: triange condition failed',ckVm2(1)));
                testCase.verifyThat(ckVm2(2),IsLessThan(1e-15),...
                    sprintf('cx=%g: triange condition failed',ckVm2(2)));
            end
            
        end

        % Test for construction with order 3 3
        function testConstructorWithDec22Ord33Vm2(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 3 3 ];
            ang = (2*pi*rand(1,8)-1);
            mus = 2*round(rand(2,8))-1;
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*
            testCase.lppufb = LpPuFb2dVm2System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % lenx3x, lenx3y
            lenx3x = get(testCase.lppufb,'lenx3x');
            lenx3y = get(testCase.lppufb,'lenx3y');
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(coefActual,dimExpctd);
            testCase.verifyThat(lenx3x,IsLessThan(2));
            testCase.verifyThat(lenx3y,IsLessThan(2));
            
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
            for iSubband = 2:prod(dec)
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end

            % Check Vm2
            ckVm2 = checkVm2(testCase.lppufb);
            testCase.verifyThat(ckVm2(1),IsLessThan(1e-15),...
                sprintf('cy=%g: triange condition failed',ckVm2(1)));
            testCase.verifyThat(ckVm2(2),IsLessThan(1e-15),...
                sprintf('cx=%g: triange condition failed',ckVm2(2)));
            
        end
        
        % Test for construction with order 3 3
        function testConstructorWithDec44Ord33Vm2(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 3 3 ];
            ang = 2*pi*rand(28,8);
            mus = 2*round(rand(8,8))-1;
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlot.*
            testCase.lppufb = LpPuFb2dVm2System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % lenx3x, lenx3y
            lenx3x = get(testCase.lppufb,'lenx3x');
            lenx3y = get(testCase.lppufb,'lenx3y');
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(coefActual,dimExpctd);
            testCase.verifyThat(lenx3x,IsLessThan(2));
            testCase.verifyThat(lenx3y,IsLessThan(2));            
            
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
            for iSubband = 2:prod(dec)
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end

            % Check Vm2
            if get(testCase.lppufb,'lambdaxueq') <=0 && ...
                    get(testCase.lppufb,'lambdayueq') <=0
                ckVm2 = checkVm2(testCase.lppufb);
                testCase.verifyThat(ckVm2(1),IsLessThan(1e-15),...
                    sprintf('cy=%g: triange condition failed',ckVm2(1)));
                testCase.verifyThat(ckVm2(2),IsLessThan(1e-15),...
                    sprintf('cx=%g: triange condition failed',ckVm2(2)));
            end
            
        end
        
    end
    
end
