classdef CplxOvsdLpPuFb3dTypeIIVm1SystemTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB3dTYPEIIVM1SYSTEMTESTCASE Test case for CplxOvsdLpPuFb3dTypeIIVm1System
    %
    % SVN identifier:
    % $Id: CplxOvsdLpPuFb3dTypeIIVm1SystemTestCase.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
    %
    properties (Constant)
        %{
            F00x = idct2([1 0; 0 0]);
            F10x = idct2([0 0; 1 0]);
            F01x = idct2([0 1; 0 0]);
            F11x = idct2([0 0; 0 1]);
            Fxx0 = permute(idct([1 ; 0]),[2 3 1]);
            Fxx1 = permute(idct([0 ; 1]),[2 3 1]);
            F000 = convn(F00x,Fxx0);
            F001 = convn(F00x,Fxx1);
            F100 = convn(F10x,Fxx0);
            F101 = convn(F10x,Fxx1);
            F010 = convn(F01x,Fxx0);
            F011 = convn(F01x,Fxx1);
            F110 = convn(F11x,Fxx0);
            F111 = convn(F11x,Fxx1);
            matrixE0 = flip([ ...
                F000(:).'
                F011(:).'
                F110(:).'
                F101(:).'
                F001(:).'
                F010(:).'
                F111(:).'
                F100(:).' ],2);
        %}
        matrixE0 = 1/(2*sqrt(2))*[
             1 ,  1 ,  1 ,  1 ,  1 ,  1 ,  1 ,  1 ;
            -1i, -1i, -1i, -1i,  1i,  1i,  1i,  1i;
            -1i, -1i,  1i,  1i, -1i, -1i,  1i,  1i;
            -1 , -1 ,  1 ,  1 ,  1 ,  1 , -1 , -1 ;
            -1i,  1i, -1i,  1i, -1i,  1i, -1i,  1i;
            -1 ,  1 , -1 ,  1 ,  1 , -1 ,  1 , -1 ;
            -1 ,  1 ,  1 , -1 , -1 ,  1 ,  1 , -1 ;
             1i, -1i, -1i,  1i, -1i,  1i,  1i, -1i];
    end
    
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
                testCase.matrixE0;
                zeros(1,8);
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System();
            
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
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System();
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
        function testConstructorWithDec222Ch9Ord000(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 0 0 0 ];
            
            % Expected values
             coefExpctd(:,:,1,1,1) =  [ 
                testCase.matrixE0;
                zeros(1,8);
            ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 2 2 2
        function testConstructorWithDec222Ch9Ord222Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(36+3*36,1);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction with order 2 2 2
        function testConstructorWithDec33Ord222(testCase)
            
            % Parameters
            dec = [ 3 3 3 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(27*13+3*(13*12+14*13+2*6),1);
            
            % Expected values
            nDec = prod(dec(1:3));
            nChs = nDec;
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
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
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec222Ch11Ord000(testCase)
            
            % Parameters
            decch = [ 2 2 2 11 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            dimExpctd = [11 8];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            
            % Check tightness
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec222Ch9Ord000Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 0 0 0 ];
            angV0 = 2*pi*rand(36,1);
            angV0(1:decch(4)) = zeros(decch(4),1);
            
            % Expected values
            import saivdr.dictionary.utility.*
            omgsV = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            matrixV0 = step(omgsV,angV0,1);
            coefExpctd(:,:,1,1,1) = ...
                matrixV0 * ...
                [
                    testCase.matrixE0;
                    zeros(1,8)
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
                
            % Actual values
            coefActual = step(testCase.lppufb,angV0,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-11,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec222Ch11Ord000Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 11 ];
            ord = [ 0 0 0 ];
            ang = 2*pi*rand(55,1);
            
            % Expected values
            dimExpctd = [11 8];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec111Ch5Ord000(testCase)
            
            % Parameters
            decch = [ 1 1 1 5 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1,1) = [...
                1;
                0;
                0;
                0;
                0];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System('DecimationFactor',decch(1:3),'NumberOfChannels',decch(4:end),'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Invalid arguments
        function testConstructorWithInvalidArguments(testCase)
            
            % Invalid input
            decch = [ 2 2 2 9 ];
            ord = [ 0 0 0 ];
            sizeInvalid = 4;
            ang = 2*pi*rand(sizeInvalid);
            
            % Expected value
            sizeExpctd = 36;
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Length of angles must be %d',...
                sizeExpctd);
            
            % Instantiation of target class
            try
                import saivdr.dictionary.cnsoltx.*
                testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                    'DecimationFactor',decch(1:3),...
                    'NumberOfChannels',decch(4:end),...
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
            decch = [ 2 2 2 9 ];
            ord = [ 0 0 0 ];
            ang = zeros(36,1);
            mus = [ 1 1 1 1 -1 -1 -1 -1 -1 ].';
            
            % Expected values
            coefExpctd(:,:,1,1,1) = [
                testCase.matrixE0(1:4,:);
               -testCase.matrixE0(5:8,:);
                zeros(1,8);
               ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 2 2
        function testConstructorWithDec222Ch9Ord222Ang0(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 2 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(9,8,3,3,3);
            coefExpctd(:,:,2,2,2) = [
                testCase.matrixE0;
                zeros(1,8)
               ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 4 4
        function testConstructorWithDec222Ch9Ord444Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 4 4 4 ];
            ang = 2*pi*rand(36+6*(12+20+4),1);
            
            % Expected values
            nDec = prod(decch(1:3));
            dimExpctd = [decch(4) nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test dec 2 2 2 order 2 2 2
        function testConstructorWithDec222Ch11Ord222Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 11 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(55+3*(20+30+4),1);
            
            % Expected values
            nDec = prod(decch(1:3));
            nChs = decch(4);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test: dec 3 3 3 order 4 4 4
        function testConstructorWithDec333Ch27Ord444Ang(testCase)
            
            % Parameters
            decch = [ 3 3 3 27 ];
            ord = [ 4 4 4 ];
            ang = 2*pi*rand(27*13+6*(13*12+14*13+12),1);
            
            % Expected values
            nDec = prod(decch(1:3));
            nChs = decch(4);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for angle setting
        function testSetAngles(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 0 0 0 ];
            angPre = pi/4*ones(36,1);
            angPst = zeros(36,1);
            
            % Expected values
            coefExpctd(:,:,1,1,1) = [
                testCase.matrixE0;
                zeros(1,8);
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,angPre,[]);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThanOrEqualTo;
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThanOrEqualTo(1e-14),sprintf('%g',coefDist));
            
            % Actual values
            coefActual = step(testCase.lppufb,angPst,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for mus setting
        function testSetMus(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 0 0 0];
            ang = zeros(36,1);
            musPre = [ 1 -1  1 -1 1 -1 1 -1 1].';
            musPst = 1;
            
            % Expected values
            coefExpctd(:,:,1,1,1) = [
                testCase.matrixE0;
                zeros(1,8);
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for subsref
        function testAnalysisFilterAt(testCase)
            
            % Expected value
            anFiltExpctd1(:,:,1) = 1/(2*sqrt(2)) * [
                1   1 ;
                1   1
                ]; 
            anFiltExpctd1(:,:,2) = 1/(2*sqrt(2)) * [
                1   1 ; 
                1   1
                ];             
            anFiltExpctd2(:,:,1) = 1/(2*sqrt(2)) * [
               -1i -1i ;
               -1i -1i
                ];                
            anFiltExpctd2(:,:,2) = 1/(2*sqrt(2)) * [
                1i  1i ;  
                1i  1i
                ];                            
            anFiltExpctd3(:,:,1) = 1/(2*sqrt(2)) * [
               -1i  1i
               -1i  1i
                ];                
            anFiltExpctd3(:,:,2) = 1/(2*sqrt(2)) * [
               -1i  1i
               -1i  1i
                ];                            
            anFiltExpctd4(:,:,1) = 1/(2*sqrt(2)) * [
               -1   1
               -1   1
                ];                
            anFiltExpctd4(:,:,2) = 1/(2*sqrt(2)) * [
                1  -1
                1  -1
                ];                            
            anFiltExpctd5(:,:,1) = 1/(2*sqrt(2)) * [
               -1i -1i
                1i  1i
                ];                
            anFiltExpctd5(:,:,2) = 1/(2*sqrt(2)) * [
               -1i -1i
                1i  1i
                ];                            
            anFiltExpctd6(:,:,1) = 1/(2*sqrt(2)) * [
               -1  -1
                1   1
                ];                
            anFiltExpctd6(:,:,2) = 1/(2*sqrt(2)) * [
                1   1
               -1  -1 
                ];                            
            anFiltExpctd7(:,:,1) = 1/(2*sqrt(2)) * [
               -1   1
                1  -1
                ];                
            anFiltExpctd7(:,:,2) = 1/(2*sqrt(2)) * [
               -1   1
                1  -1 
                ];                
            anFiltExpctd8(:,:,1) = 1/(2*sqrt(2)) * [
                1i -1i
               -1i  1i
                ];
            anFiltExpctd8(:,:,2) = 1/(2*sqrt(2)) * [
               -1i  1i
                1i -1i
                ];
            anFiltExpctd9 = zeros(2,2,2);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm0System(...
                'OutputMode','AnalysisFilterAt');
            
            % Actual values
            anFiltActual1 = step(testCase.lppufb,[],[],1);
            anFiltActual2 = step(testCase.lppufb,[],[],2);
            anFiltActual3 = step(testCase.lppufb,[],[],3);
            anFiltActual4 = step(testCase.lppufb,[],[],4);
            anFiltActual5 = step(testCase.lppufb,[],[],5);
            anFiltActual6 = step(testCase.lppufb,[],[],6);
            anFiltActual7 = step(testCase.lppufb,[],[],7);
            anFiltActual8 = step(testCase.lppufb,[],[],8);
            anFiltActual9 = step(testCase.lppufb,[],[],9);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(anFiltActual1,size(anFiltExpctd1));
            testCase.verifySize(anFiltActual2,size(anFiltExpctd2));            
            testCase.verifySize(anFiltActual3,size(anFiltExpctd3));
            testCase.verifySize(anFiltActual4,size(anFiltExpctd4));                        
            testCase.verifySize(anFiltActual5,size(anFiltExpctd5));
            testCase.verifySize(anFiltActual6,size(anFiltExpctd6));            
            testCase.verifySize(anFiltActual7,size(anFiltExpctd7));
            testCase.verifySize(anFiltActual8,size(anFiltExpctd8));                                    
            testCase.verifySize(anFiltActual9,size(anFiltExpctd9));                                    
            
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
            dist = norm(anFiltExpctd6(:)-anFiltActual6(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd7(:)-anFiltActual7(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd8(:)-anFiltActual8(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));                
            dist = norm(anFiltExpctd9(:)-anFiltActual9(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));                            
            
            
        end
    function testAnalysisFilters(testCase)
            
            % Expected value
            anFiltExpctd1(:,:,1) = 1/(2*sqrt(2)) * [
                1   1 ;
                1   1
                ]; 
            anFiltExpctd1(:,:,2) = 1/(2*sqrt(2)) * [
                1   1 ; 
                1   1
                ];             
            anFiltExpctd2(:,:,1) = 1/(2*sqrt(2)) * [
               -1i -1i ;
               -1i -1i
                ];                
            anFiltExpctd2(:,:,2) = 1/(2*sqrt(2)) * [
                1i  1i ;  
                1i  1i
                ];                            
            anFiltExpctd3(:,:,1) = 1/(2*sqrt(2)) * [
               -1i  1i
               -1i  1i
                ];                
            anFiltExpctd3(:,:,2) = 1/(2*sqrt(2)) * [
               -1i  1i
               -1i  1i
                ];                            
            anFiltExpctd4(:,:,1) = 1/(2*sqrt(2)) * [
               -1   1
               -1   1
                ];                
            anFiltExpctd4(:,:,2) = 1/(2*sqrt(2)) * [
                1  -1
                1  -1
                ];                            
            anFiltExpctd5(:,:,1) = 1/(2*sqrt(2)) * [
               -1i -1i
                1i  1i
                ];                
            anFiltExpctd5(:,:,2) = 1/(2*sqrt(2)) * [
               -1i -1i
                1i  1i
                ];                            
            anFiltExpctd6(:,:,1) = 1/(2*sqrt(2)) * [
               -1  -1
                1   1
                ];                
            anFiltExpctd6(:,:,2) = 1/(2*sqrt(2)) * [
                1   1
               -1  -1 
                ];                            
            anFiltExpctd7(:,:,1) = 1/(2*sqrt(2)) * [
               -1   1
                1  -1
                ];                
            anFiltExpctd7(:,:,2) = 1/(2*sqrt(2)) * [
               -1   1
                1  -1 
                ];                
            anFiltExpctd8(:,:,1) = 1/(2*sqrt(2)) * [
                1i -1i
               -1i  1i
                ];
            anFiltExpctd8(:,:,2) = 1/(2*sqrt(2)) * [
               -1i  1i
                1i -1i
                ];
            anFiltExpctd9 = zeros(2,2,2);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm0System(...
                'OutputMode','AnalysisFilters');
            
            % Actual values
            anFiltsActual = step(testCase.lppufb,[],[]);
            anFiltActual1 = anFiltsActual(:,:,:,1);
            anFiltActual2 = anFiltsActual(:,:,:,2);
            anFiltActual3 = anFiltsActual(:,:,:,3);
            anFiltActual4 = anFiltsActual(:,:,:,4);
            anFiltActual5 = anFiltsActual(:,:,:,5);
            anFiltActual6 = anFiltsActual(:,:,:,6);
            anFiltActual7 = anFiltsActual(:,:,:,7);
            anFiltActual8 = anFiltsActual(:,:,:,8);
            anFiltActual9 = anFiltsActual(:,:,:,9);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(anFiltActual1,size(anFiltExpctd1));
            testCase.verifySize(anFiltActual2,size(anFiltExpctd2));
            testCase.verifySize(anFiltActual3,size(anFiltExpctd3));
            testCase.verifySize(anFiltActual4,size(anFiltExpctd4));
            testCase.verifySize(anFiltActual5,size(anFiltExpctd5));
            testCase.verifySize(anFiltActual6,size(anFiltExpctd6));
            testCase.verifySize(anFiltActual7,size(anFiltExpctd7));
            testCase.verifySize(anFiltActual8,size(anFiltExpctd8));
            testCase.verifySize(anFiltActual9,size(anFiltExpctd9));
            
            dist = max(abs(anFiltExpctd1(:)-anFiltActual1(:))./abs(anFiltExpctd1(:)));
            testCase.verifyEqual(anFiltActual1,anFiltExpctd1,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd2(:)-anFiltActual2(:))./abs(anFiltExpctd2(:)));
            testCase.verifyEqual(anFiltActual2,anFiltExpctd2,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd3(:)-anFiltActual3(:))./abs(anFiltExpctd3(:)));
            testCase.verifyEqual(anFiltActual3,anFiltExpctd3,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd4(:)-anFiltActual4(:))./abs(anFiltExpctd4(:)));
            testCase.verifyEqual(anFiltActual4,anFiltExpctd4,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd5(:)-anFiltActual5(:))./abs(anFiltExpctd5(:)));
            testCase.verifyEqual(anFiltActual5,anFiltExpctd5,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd6(:)-anFiltActual6(:))./abs(anFiltExpctd6(:)));
            testCase.verifyEqual(anFiltActual6,anFiltExpctd6,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd7(:)-anFiltActual7(:))./abs(anFiltExpctd7(:)));
            testCase.verifyEqual(anFiltActual7,anFiltExpctd7,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd8(:)-anFiltActual8(:))./abs(anFiltExpctd8(:)));
            testCase.verifyEqual(anFiltActual8,anFiltExpctd8,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd9(:)-anFiltActual9(:))./abs(anFiltExpctd9(:)));
            testCase.verifyEqual(anFiltActual8,anFiltExpctd8,'RelTol',1e-15,sprintf('%g',dist));
            
        end

        % Test dec 2 2 2 ch 9 order 0 0 2
        function testConstructorWithDec222Ch9Ord002(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 0 0 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(9,8,1,1,3);
            coefExpctd(:,:,1,1,2) = [
                testCase.matrixE0;
                zeros(1,8);
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test dec 1 1 1 ch 5 order 0 0 2
        function testConstructorWithDec11Ch5Ord02(testCase)
            
            % Parameters
            decch = [ 1 1 1 5 ];
            ord = [ 0 0 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(5,1,1,1,3);
            coefExpctd(:,:,1,1,2) = [
                1 ;
                0 ;
                0 ;
                0 ;
                0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test dec 2 2 2 order 4 4 4
        function testConstructorWithDec222Ch9Ord444(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 4 4 4 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(9,8,5,5,5);
            coefExpctd(:,:,3,3,3) = [
                testCase.matrixE0;
                zeros(1,8);
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
 
        % Test dec 2 2 2 ch 9 order 6 6 6
        function testConstructorWithDec222Ch9Ord666Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 6 6 6 ];
            ang = 2*pi*rand(36+9*(12+20+4),1);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end

        % Test dec 2 2 2 ch 11 order 2 2 2
        function testConstructorWithDec222Ch11Ord444Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 11 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(55+3*(20+30+4),1);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end

        % Test dec 3 3 3 ch 29 order 2 2 2
        function testConstructorWithDec333Ch29Ord222Ang(testCase)
            
            % Parameters
            decch = [ 3 3 3 29 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(29*14+3*(14*13+15*14+14),1);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end

        % Test dec 1 1 1 ch 5 order 2 2 2
        function testConstructorWithDec112Ch5Ord232Ang(testCase)
            
            % Parameters
            decch = [ 1 1 1 5 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(10+3*(2+6+2),1);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end

        % Test for construction with order 2 2
        function testConstructorWithDec222Ch54Ord222(testCase)
            
            % Parameters
            decch = [ 2 2 2 5 4 ];
            ord = [ 2 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(9,8,3,3,3);
            coefExpctd(:,:,2,2,2) = [
                testCase.matrixE0;
                zeros(1,8);
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test dec 2 2 2 order 4 4 4
        function testConstructorWithDec222Ch54Ord222Ang(testCase)
            
          % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(36+3*(12+20+4),1);
            
            % Expected values
            nDec = prod(decch(1:3));
            nChs = sum(decch(4));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec222Ch9Ord222AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(36+3*(12+20+4),1);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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

        % Test for construction with order 2 2 2
        function testConstructorWithDec222Ch9Ord444AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 4 4 4 ];
            ang = 2*pi*rand(36+6*(12+20+4),1);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            mstab = [ 9 9 ];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixContainer
            paramExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(9),1);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end

%         % Test for construction with order 2 2
%         function testParameterMatrixSetRandAngWithDec222Ch54Ord222(testCase)
%             
%             % Parameters
%             decch = [ 2 2 2 5 4 ];
%             ord = [ 2 2 2 ];
%             mstab = [ 5 5 ; 4 4 ; 5 5 ; 4 4 ; 5 5 ; 4 4 ; 5 5 ; 4 4  ];
%             
%             % Expected values
%             import saivdr.dictionary.utility.*
%             paramMtxExpctd = ParameterMatrixContainer(...
%                 'MatrixSizeTable',mstab);
%             step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
%             step(paramMtxExpctd,-eye(mstab(2,:)),uint32(2)); % U0
%             step(paramMtxExpctd, eye(mstab(3,:)),uint32(3)); % Wz1
%             step(paramMtxExpctd,-eye(mstab(4,:)),uint32(4)); % Uz1            
%             step(paramMtxExpctd, eye(mstab(5,:)),uint32(5)); % Wx1
%             step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % Ux1
%             step(paramMtxExpctd, eye(mstab(7,:)),uint32(7)); % Wy1
%             step(paramMtxExpctd,-eye(mstab(8,:)),uint32(8)); % Uy1
%             coefExpctd = get(paramMtxExpctd,'Coefficients');
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb3dTypeIIVm1System(...
%                 'DecimationFactor',decch(1:3),...
%                 'NumberOfChannels',decch(4:end),...
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
%             Wz1 = step(paramMtxActual,[],uint32(3));
%             Wx1 = step(paramMtxActual,[],uint32(5));
%             Wy1 = step(paramMtxActual,[],uint32(7));
%             G = Wy1*Wx1*Wz1*W0;
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
%             for iSubband = 2:sum(decch(4:5))
%                 H = step(testCase.lppufb,[],[],iSubband);
%                 dc = abs(sum(H(:)));
%                 testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
%             end            
%         end                   
        
    end
    
end
