classdef OvsdLpPuFb3dTypeIVm1SystemTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB3dTYPEIVM1TESTCASE Test case for OvsdLpPuFb3dTypeIVm1System
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
            1     1     1     1     1     1     1     1
            1     1    -1    -1    -1    -1     1     1
            1    -1    -1     1     1    -1    -1     1
            1    -1     1    -1    -1     1    -1     1
            -1    -1    -1    -1     1     1     1     1
            -1    -1     1     1    -1    -1     1     1
            -1     1     1    -1     1    -1    -1     1
            -1     1    -1     1    -1     1    -1     1
            ];
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
            
            % Expected values yxz
            coefExpctd(:,:,1,1,1) = testCase.matrixE0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System();
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,...
                sprintf('%g',coefDist));
            
        end
    
        % Test for default construction
        function testConstructorWithDeepCopy(testCase)
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System();
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
        function testConstructorWithOrd000(testCase)
            
            % Parameters
            dec = [ 2 2 2 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1,1) = testCase.matrixE0;

            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,...
                sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec444Ord000(testCase)
            
            % Parameters
            dec = [ 4 4 4 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            dimExpctd = [64 64];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
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
        function testConstructorWithDec22Ch4Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 2 8 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1,1) = testCase.matrixE0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,...
                sprintf('%g',coefDist));

        end

         % Test for construction with order 2 0
        function testConstructorWithDec222Ch10Ord200(testCase)
            
            % Parameters
            decch = [ 2 2 2 10 ];
            ord = [ 2 0 0 ];
            ang = 2*pi*rand(10,4);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1]; 
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyTrue(coefDist<1e-15,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 0 2 0
        function testConstructorWithDec222Ch10Ord020(testCase)
            
            % Parameters
            decch = [ 2 2 2 10 ];
            ord = [ 0 2 0 ];
            ang = 2*pi*rand(10,4);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyTrue(coefDist<1e-15,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 0 0 2
        function testConstructorWithDec222Ch10Ord002(testCase)
            
            % Parameters
            decch = [ 2 2 2 10 ];
            ord = [ 0 0 2 ];
            ang = 2*pi*rand(10,4);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyTrue(coefDist<1e-15,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 0 2
        function testConstructorWithDec222Ch8Ord222(testCase)
            
            % Parameters
            decch = [ 2 2 2 10 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(10,8);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyTrue(coefDist<1e-15,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec222Ch12Ord000(testCase)
            
            % Parameters
            decch = [ 2 2 2 12 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            dimExpctd = [12 8];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
        function testConstructorWithDec222Ch10Ord000(testCase)
            
            % Parameters
            decch = [ 2 2 2 10 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            dimExpctd = [10 8];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
        function testConstructorWithOrd000Ang(testCase)
            
            % Parameters
            dec = [ 2 2 2 ];
            ord = [ 0 0 0 ];
            ang = zeros(6,2);
            
            % Expected values
            coefExpctd(:,:,1,1,1) = testCase.matrixE0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./...
                (abs(coefExpctd(:))));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15,...
                sprintf('%g',coefDist));
        end
        
        % Test dec 2 2 ch 8 order 4 4
        function testConstructorWithDec222Ch10Ord444Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 10 ];
            ord = [ 4 4 4 ];
            ang = 2*pi*rand(10,14);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyTrue(coefDist<1e-15,sprintf('%g',coefDist));
            
        end
          
        % Test dec 2 2 2 ch 10 order 2 2 2
        function testConstructorWithDec222Ch10Ord222AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 2 2 2 10 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(10,8);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
        
        % Test dec 2 2 2 ch 10 order 4 4 4
        function testConstructorWithDec222Ch10Ord444AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 2 2 2 10 ];
            ord = [ 4 4 4 ];
            ang = 2*pi*rand(10,14);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
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

        % Test dec 2 2 2 ch 4 4 order 4 4 4
        function testConstructorWithDec222Ch44Ord222AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 2 2 2 4 4 ];
            ord = [ 4 4 4 ];
            ang = 2*pi*rand(6,14);
            
            % Expected values
            nChs = sum(decch(4:5));
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
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
        function testParameterMatrixContainer(testCase)
            
            % Preparation
            mstab = [ 4 4 ; 4 4 ];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixContainer
            paramExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(4),1);
            step(paramExpctd,eye(4),2);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end

        % Test for construction with order 2 2 2
        function testParameterMatrixSetRandAngMuWithDec222Ch44Ord222(testCase)
            
            % Parameters
            decch = [ 2 2 2 4 4 ];
            ord = [ 2 2 2 ];
            mstab = 4*ones(8,2);
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
            step(paramMtxExpctd,-eye(mstab(3,:)),uint32(3)); % Uz1
            step(paramMtxExpctd,-eye(mstab(4,:)),uint32(4)); % Uz2
            step(paramMtxExpctd,-eye(mstab(5,:)),uint32(5)); % Ux1
            step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % Ux2
            step(paramMtxExpctd,-eye(mstab(7,:)),uint32(7)); % Uy1
            step(paramMtxExpctd,-eye(mstab(8,:)),uint32(8)); % Uy2            
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            W0  = step(omgs, [zeros(3,1); ang(4:end,1)], [1; mus(2:end,1)]);
            U0  = step(omgs, ang(:,2), mus(:,2));
            Uz1 = step(omgs, ang(:,3), mus(:,3));
            Uz2 = step(omgs, ang(:,4), mus(:,4));
            Ux1 = step(omgs, ang(:,5), mus(:,5));
            Ux2 = step(omgs, ang(:,6), mus(:,6));
            Uy1 = step(omgs, ang(:,7), mus(:,7));
            Uy2 = step(omgs, ang(:,8), mus(:,8));            
            step(paramMtxExpctd,W0 ,uint32(1)); % W0
            step(paramMtxExpctd,U0 ,uint32(2)); % U0
            step(paramMtxExpctd,Uz1,uint32(3)); % Uz1
            step(paramMtxExpctd,Uz2,uint32(4)); % Uz2            
            step(paramMtxExpctd,Ux1,uint32(5)); % Ux1
            step(paramMtxExpctd,Ux2,uint32(6)); % Ux2
            step(paramMtxExpctd,Uy1,uint32(7)); % Uy1            
            step(paramMtxExpctd,Uy2,uint32(8)); % Uy2
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
            for iSubband = 2:sum(decch(4:5))
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
            
        end
         
        % Test for construction with order 2 2 2
        function testParameterMatrixSetRandAngWithDec222Ch44Ord222(testCase)

            % Parameters
            decch = [ 2 2 2 4 4 ];
            ord = [ 2 2 2 ];
            mstab = 4*ones(8,2);
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
            step(paramMtxExpctd,-eye(mstab(3,:)),uint32(3)); % Uz1
            step(paramMtxExpctd,-eye(mstab(4,:)),uint32(4)); % Uz2            
            step(paramMtxExpctd,-eye(mstab(6,:)),uint32(5)); % Ux1
            step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % Ux2
            step(paramMtxExpctd,-eye(mstab(7,:)),uint32(7)); % Uy1
            step(paramMtxExpctd,-eye(mstab(8,:)),uint32(8)); % Uy2
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            W0  = step(omgs, [zeros(3,1); ang(4:end,1)], [1; mus(2:end,1)]);
            U0  = step(omgs, ang(:,2), mus(:,2));
            Uz1 = step(omgs, ang(:,3), mus(:,3));
            Uz2 = step(omgs, ang(:,4), mus(:,4));
            Ux1 = step(omgs, ang(:,5), mus(:,5));
            Ux2 = step(omgs, ang(:,6), mus(:,6));            
            Uy1 = step(omgs, ang(:,7), mus(:,7));
            Uy2 = step(omgs, ang(:,8), mus(:,8));
            step(paramMtxExpctd,W0 ,uint32(1)); % W0
            step(paramMtxExpctd,U0 ,uint32(2)); % U0
            step(paramMtxExpctd,Uz1,uint32(3)); % Uz1
            step(paramMtxExpctd,Uz2,uint32(4)); % Uz2            
            step(paramMtxExpctd,Ux1,uint32(5)); % Ux1
            step(paramMtxExpctd,Ux2,uint32(6)); % Ux2
            step(paramMtxExpctd,Uy1,uint32(7)); % Uy1            
            step(paramMtxExpctd,Uy2,uint32(8)); % Uy2
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
            for iSubband = 2:sum(decch(4:5))
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end            
        end  

        % Test for construction with order 2 2
        function testParameterMatrixSetRandMuWithDec222Ch44Ord222(testCase)
            
            % Parameters
            decch = [ 2 2 2 4 4];
            ord = [ 2 2 2 ];
            mstab = 4*ones(8,2);
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
            step(paramMtxExpctd,-eye(mstab(3,:)),uint32(3)); % Uz1
            step(paramMtxExpctd,-eye(mstab(4,:)),uint32(4)); % Uz2
            step(paramMtxExpctd,-eye(mstab(5,:)),uint32(5)); % Ux1
            step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % Ux2            
            step(paramMtxExpctd,-eye(mstab(7,:)),uint32(7)); % Uy1
            step(paramMtxExpctd,-eye(mstab(8,:)),uint32(8)); % Uy2
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm1System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            W0  = step(omgs,[zeros(3,1); ang(4:end,1)], [1; mus(2:end,1)]);
            U0  = step(omgs,ang(:,2), mus(:,2));
            Uz1 = step(omgs,ang(:,3), mus(:,3));
            Uz2 = step(omgs,ang(:,4), mus(:,4));
            Ux1 = step(omgs,ang(:,5), mus(:,5));
            Ux2 = step(omgs,ang(:,6), mus(:,6));            
            Uy1 = step(omgs,ang(:,7), mus(:,7));
            Uy2 = step(omgs,ang(:,8), mus(:,8));
            step(paramMtxExpctd,W0 ,uint32(1)); % W0
            step(paramMtxExpctd,U0 ,uint32(2)); % U0
            step(paramMtxExpctd,Uz1,uint32(3)); % Uz1
            step(paramMtxExpctd,Uz2,uint32(4)); % Uz2
            step(paramMtxExpctd,Ux1,uint32(5)); % Ux1
            step(paramMtxExpctd,Ux2,uint32(6)); % Ux2            
            step(paramMtxExpctd,Uy1,uint32(7)); % Uy1            
            step(paramMtxExpctd,Uy2,uint32(8)); % Uy2
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
            for iSubband = 2:sum(decch(4:5))
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end
        end  

    end
    
end
