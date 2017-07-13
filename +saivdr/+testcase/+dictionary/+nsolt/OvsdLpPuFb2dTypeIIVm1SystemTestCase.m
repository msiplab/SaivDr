classdef OvsdLpPuFb2dTypeIIVm1SystemTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB2DTYPEIIVM1SYSTEMTESTCASE Test case for OvsdLpPuFb2dTypeIIVm1System
    %
    % SVN identifier:
    % $Id: OvsdLpPuFb2dTypeIIVm1SystemTestCase.m 683 2015-05-29 08:22:13Z sho $
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
            delete(testCase.lppufb);
        end
    end
    
    methods (Test)
        
        % Test for default construction
        function testConstructor(testCase)
            
            % Expected values
            coefExpctd = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System();
            
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
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System();
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
        function testConstructorWithDec33Ord00(testCase)
            
            % Parameters
            dec = [ 3 3 ];
            ord = [ 0 0 ];
            
            % Expected values
            C = dctmtx(3);
            coefExpctd(1,:,1,1) = reshape(C(1,:).'*C(1,:),1,9);
            coefExpctd(2,:,1,1) = reshape(C(3,:).'*C(1,:),1,9);
            coefExpctd(3,:,1,1) = reshape(C(1,:).'*C(3,:),1,9);
            coefExpctd(4,:,1,1) = reshape(C(3,:).'*C(3,:),1,9);
            coefExpctd(5,:,1,1) = reshape(C(2,:).'*C(2,:),1,9);
            coefExpctd(6,:,1,1) = -reshape(C(2,:).'*C(1,:),1,9);
            coefExpctd(7,:,1,1) = -reshape(C(2,:).'*C(3,:),1,9);
            coefExpctd(8,:,1,1) = -reshape(C(1,:).'*C(2,:),1,9);
            coefExpctd(9,:,1,1) = -reshape(C(3,:).'*C(2,:),1,9);
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System('DecimationFactor',dec,'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch5Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System('DecimationFactor',decch(1:2),'NumberOfChannels',decch(3:end),'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 0 2
        function testConstructorWithDec22Ch5Ord02Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 2 ];
            ang = 2*pi*rand(4,2);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 0
        function testConstructorWithDec22Ch5Ord20Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 2 0 ];
            ang = 2*pi*rand(4,2);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch5Ord22Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(4,3);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 0 2
        function testConstructorWithDec33Ord02(testCase)
            
            % Parameters
            dec = [ 3 3 ];
            ord = [ 0 2 ];
            ang = 2*pi*rand(16,2);
            
            % Expected values
            nDec = dec(1)*dec(2);
            nChs = nDec;
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 0
        function testConstructorWithDec33Ord20(testCase)
            
            % Parameters
            dec = [ 3 3 ];
            ord = [ 2 0 ];
            ang = 2*pi*rand(16,2);
            
            % Expected values
            nDec = dec(1)*dec(2);
            nChs = nDec;
            dimExpctd = [nChs nDec ord(1)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec33Ord22(testCase)
            
            % Parameters
            dec = [ 3 3 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(16,3);
            
            % Expected values
            nDec = dec(1)*dec(2);
            nChs = nDec;
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch7Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 7 ];
            ord = [ 0 0 ];
            
            % Expected values
            dimExpctd = [7 4];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System('DecimationFactor',decch(1:2),'NumberOfChannels',decch(3:end),'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch9Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 9 ];
            ord = [ 0 0 ];
            
            % Expected values
            dimExpctd = [9 4];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System('DecimationFactor',decch(1:2),'NumberOfChannels',decch(3:end),'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec33Ch9Ord00Ang0(testCase)
            
            % Parameters
            decch = [ 3 3 9 ];
            ord = [ 0 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333;0.235702260395516,-0.471404520791032,0.235702260395516,0.235702260395516,-0.471404520791032,0.235702260395516,0.235702260395516,-0.471404520791032,0.235702260395516;0.235702260395516,0.235702260395516,0.235702260395516,-0.471404520791032,-0.471404520791032,-0.471404520791032,0.235702260395516,0.235702260395516,0.235702260395516;0.166666666666667,-0.333333333333333,0.166666666666667,-0.333333333333333,0.666666666666667,-0.333333333333333,0.166666666666667,-0.333333333333333,0.166666666666667;0.500000000000000,3.53525079574969e-17,-0.500000000000000,3.53525079574969e-17,2.49959963776976e-33,-3.53525079574969e-17,-0.500000000000000,-3.53525079574969e-17,0.500000000000000;-0.408248290463863,-2.88652018745164e-17,0.408248290463863,-0.408248290463863,-2.88652018745164e-17,0.408248290463863,-0.408248290463863,-2.88652018745164e-17,0.408248290463863;-0.288675134594813,-2.04107799857892e-17,0.288675134594813,0.577350269189626,4.08215599715784e-17,-0.577350269189626,-0.288675134594813,-2.04107799857892e-17,0.288675134594813;-0.408248290463863,-0.408248290463863,-0.408248290463863,-2.88652018745164e-17,-2.88652018745164e-17,-2.88652018745164e-17,0.408248290463863,0.408248290463863,0.408248290463863;-0.288675134594813,0.577350269189626,-0.288675134594813,-2.04107799857892e-17,4.08215599715784e-17,-2.04107799857892e-17,0.288675134594813,-0.577350269189626,0.288675134594813;];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec33Ch9Ord00Ang(testCase)
            
            % Parameters
            decch = [ 3 3 9 ];
            ord = [ 0 0 ];
            angW = zeros(10,1);
            angU = 2*pi*rand(6,1);
            
            % Expected values
            import saivdr.dictionary.utility.*
            omgsW = OrthonormalMatrixGenerationSystem();
            omgsU = OrthonormalMatrixGenerationSystem();
            matrixW0 = step(omgsW,angW,1);
            matrixU0 = step(omgsU,angU,1);
            coefExpctd(:,:,1,1) = ...
                blkdiag(matrixW0, matrixU0) * ...
                [0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333;0.235702260395516,-0.471404520791032,0.235702260395516,0.235702260395516,-0.471404520791032,0.235702260395516,0.235702260395516,-0.471404520791032,0.235702260395516;0.235702260395516,0.235702260395516,0.235702260395516,-0.471404520791032,-0.471404520791032,-0.471404520791032,0.235702260395516,0.235702260395516,0.235702260395516;0.166666666666667,-0.333333333333333,0.166666666666667,-0.333333333333333,0.666666666666667,-0.333333333333333,0.166666666666667,-0.333333333333333,0.166666666666667;0.500000000000000,3.53525079574969e-17,-0.500000000000000,3.53525079574969e-17,2.49959963776976e-33,-3.53525079574969e-17,-0.500000000000000,-3.53525079574969e-17,0.500000000000000;-0.408248290463863,-2.88652018745164e-17,0.408248290463863,-0.408248290463863,-2.88652018745164e-17,0.408248290463863,-0.408248290463863,-2.88652018745164e-17,0.408248290463863;-0.288675134594813,-2.04107799857892e-17,0.288675134594813,0.577350269189626,4.08215599715784e-17,-0.577350269189626,-0.288675134594813,-2.04107799857892e-17,0.288675134594813;-0.408248290463863,-0.408248290463863,-0.408248290463863,-2.88652018745164e-17,-2.88652018745164e-17,-2.88652018745164e-17,0.408248290463863,0.408248290463863,0.408248290463863;-0.288675134594813,0.577350269189626,-0.288675134594813,-2.04107799857892e-17,4.08215599715784e-17,-2.04107799857892e-17,0.288675134594813,-0.577350269189626,0.288675134594813;];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
                
            % Actual values
            coefActual = step(testCase.lppufb,[angW;angU],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-11,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch5Ord00Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 0 ];
            ang = 2*pi*rand(4,1);
            
            % Expected values
            dimExpctd = [5 4];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch7Ord00Ang(testCase)
            
            % Parameters
            decch = [ 2 2 7 ];
            ord = [ 0 0 ];
            ang = 2*pi*rand(9,1);
            
            % Expected values
            dimExpctd = [7 4];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch9Ord00Ang(testCase)
            
            % Parameters
            decch = [ 2 2 9 ];
            ord = [ 0 0 ];
            ang = 2*pi*rand(16,1);
            
            % Expected values
            dimExpctd = [9 4];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec33Ch11Ord00Ang(testCase)
            
            % Parameters
            decch = [ 3 3 11 ];
            ord = [ 0 0 ];
            ang = 2*pi*rand(25,1);
            
            % Expected values
            dimExpctd = [11 9];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec11Ch5Ord00(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = [...
                1;
                0;
                0;
                0;
                0];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System('DecimationFactor',decch(1:2),'NumberOfChannels',decch(3:end),'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec11Ch5Ord00Ang0(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 0 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                1 ;
                0 ;
                0 ;
                0 ;
                0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec11Ch5Ord00AngPi3(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 0 0 ];
            angW = zeros(3,1);
            angU = pi/3;
            
            % Expected values
            import saivdr.dictionary.utility.*
            omgsW = OrthonormalMatrixGenerationSystem();
            omgsU = OrthonormalMatrixGenerationSystem();
            matrixW0 = step(omgsW,angW,1);
            matrixU0 = step(omgsU,angU,1);
            coefExpctd(:,:,1,1) = ...
                blkdiag(matrixW0, matrixU0) * ...
                [ 1 0 0 0 0 ].';
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            decch = [ 2 2 5 ];
            ord = [ 0 0 ];
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
                import saivdr.dictionary.nsolt.*
                testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                    'DecimationFactor',decch(1:2),...
                    'NumberOfChannels',decch(3:end),...
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
            decch = [ 2 2 5 ];
            ord = [ 0 0 ];
            ang = [ 0 0 0 0 ].';
            mus = [ 1 1 1 -1 -1 ].';
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                1 -1  1 -1 ;
                1  1 -1 -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch5Ord22Ang0(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,2,1) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,3,1) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,1,2) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,2,2) = [
                0.5000    0.5000    0.5000    0.5000
                0.5000   -0.5000   -0.5000    0.5000
                0     0     0     0
                -0.5000    0.5000   -0.5000    0.5000
                -0.5000   -0.5000    0.5000    0.5000 ];
            coefExpctd(:,:,3,2) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,1,3) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,2,3) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,3,3) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4 4
        function testConstructorWithDec22Ch5Ord44Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(4,5);
            
            % Expected values
            nDecs = prod(decch(1:2));
            dimExpctd = [decch(3) nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 3 3 order 2 2
        function testConstructorWithDec33Ch9Ord22Ang0(testCase)
            
            % Parameters
            decch = [ 3 3 9 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = zeros(9);
            coefExpctd(:,:,2,1) = zeros(9);
            coefExpctd(:,:,3,1) = zeros(9);
            coefExpctd(:,:,1,2) = zeros(9);
            coefExpctd(:,:,2,2) = [0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333;0.235702260395516,-0.471404520791032,0.235702260395516,0.235702260395516,-0.471404520791032,0.235702260395516,0.235702260395516,-0.471404520791032,0.235702260395516;0.235702260395516,0.235702260395516,0.235702260395516,-0.471404520791032,-0.471404520791032,-0.471404520791032,0.235702260395516,0.235702260395516,0.235702260395516;0.166666666666667,-0.333333333333333,0.166666666666667,-0.333333333333333,0.666666666666667,-0.333333333333333,0.166666666666667,-0.333333333333333,0.166666666666667;0.500000000000000,3.53525079574969e-17,-0.500000000000000,3.53525079574969e-17,2.49959963776976e-33,-3.53525079574969e-17,-0.500000000000000,-3.53525079574969e-17,0.500000000000000;-0.408248290463863,-2.88652018745164e-17,0.408248290463863,-0.408248290463863,-2.88652018745164e-17,0.408248290463863,-0.408248290463863,-2.88652018745164e-17,0.408248290463863;-0.288675134594813,-2.04107799857892e-17,0.288675134594813,0.577350269189626,4.08215599715784e-17,-0.577350269189626,-0.288675134594813,-2.04107799857892e-17,0.288675134594813;-0.408248290463863,-0.408248290463863,-0.408248290463863,-2.88652018745164e-17,-2.88652018745164e-17,-2.88652018745164e-17,0.408248290463863,0.408248290463863,0.408248290463863;-0.288675134594813,0.577350269189626,-0.288675134594813,-2.04107799857892e-17,4.08215599715784e-17,-2.04107799857892e-17,0.288675134594813,-0.577350269189626,0.288675134594813;];
            coefExpctd(:,:,3,2) = zeros(9);
            coefExpctd(:,:,1,3) = zeros(9);
            coefExpctd(:,:,2,3) = zeros(9);
            coefExpctd(:,:,3,3) = zeros(9);
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 2 2
        function testConstructorWithDec22Ch7Ord22Ang(testCase)
            
            % Parameters
            decch = [ 2 2 7 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(9,3);
            
            % Expected values
            nDecs = prod(decch(1:2));
            nChs = decch(3);
            dimExpctd = [nChs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test: dec 3 3 order 4 4
        function testConstructorWithDec33Ch9Ord44Ang(testCase)
            
            % Parameters
            decch = [ 3 3 9 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(16,5);
            
            % Expected values
            nDecs = prod(decch(1:2));
            nChs = decch(3);
            dimExpctd = [nChs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for angle setting
        function testSetAngles(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 0 ];
            angPre = [ pi/4 pi/4 pi/4 pi/4 ].';
            angPst = [ 0 0 0 0 ].';
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0 0 0 0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            decch = [ 2 2 5 ];
            ord = [ 0 0 ];
            ang = [ 0 0 0 0 ].';
            musPre = [ 1 -1  1 -1 1].';
            musPst = 1;
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0 0 0 0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
        
        % Test for char
        function testChar(testCase)
            
            % Expected value
            charExpctd = [...
                '[', 10, ...
                9, '0.5 + 0.5*y^(-1) + 0.5*x^(-1) + 0.5*y^(-1)*x^(-1);', 10, ...
                9, '0.5 - 0.5*y^(-1) - 0.5*x^(-1) + 0.5*y^(-1)*x^(-1);', 10, ...
                9, '0;', 10, ...
                9, '-0.5 + 0.5*y^(-1) - 0.5*x^(-1) + 0.5*y^(-1)*x^(-1);', 10, ...
                9, '-0.5 - 0.5*y^(-1) + 0.5*x^(-1) + 0.5*y^(-1)*x^(-1)', 10, ...
                ']' ...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'OutputMode','Char');
            
            % Actual values
            charActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(charExpctd, charActual);
            
        end

        % Test for subsref
        function testAnalysisFilterAt(testCase)
            
            % Expected value
            anFiltExpctd1 = 1/2*[ 1 1 ;  1 1 ];
            anFiltExpctd2 = 1/2*[ 1 -1 ; -1 1 ];
            anFiltExpctd3 = [0 0 ; 0 0 ];
            anFiltExpctd4 = 1/2*[-1 -1 ;  1  1 ];
            anFiltExpctd5 = 1/2*[-1  1 ; -1  1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
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
            anFiltExpctd1 = 1/2*[ 1 1 ;  1 1 ];
            anFiltExpctd2 = 1/2*[ 1 -1 ; -1 1 ];
            anFiltExpctd3 = [0 0 ; 0 0 ];
            anFiltExpctd4 = 1/2*[-1 -1 ;  1  1 ];
            anFiltExpctd5 = 1/2*[-1  1 ; -1  1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'OutputMode','AnalysisFilters');
            
            % Actual values
            anFiltsActual = step(testCase.lppufb,[],[]);
            anFiltActual1 = anFiltsActual(:,:,1);
            anFiltActual2 = anFiltsActual(:,:,2);
            anFiltActual3 = anFiltsActual(:,:,3);
            anFiltActual4 = anFiltsActual(:,:,4);
            anFiltActual5 = anFiltsActual(:,:,5);            
            
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
        
        % Test dec 2 2 ch 5 order 0 2
        function testConstructorWithDec22Ch5Ord02(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                1 -1  1 -1 ;
                1  1 -1 -1 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 1 ch 5 order 0 2
        function testConstructorWithDec11Ch5Ord02(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 0 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  ;
                0  ;
                0  ;
                0  ;
                0  ];
            
            coefExpctd(:,:,1,2) = [
                1 ;
                0 ;
                0 ;
                0 ;
                0 ];
            
            coefExpctd(:,:,1,3) = [
                0  ;
                0  ;
                0  ;
                0  ;
                0  ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 1 ch 5 order 2 2
        function testConstructorWithDec11Ch4Ord22(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2,1) =  [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3,1) = [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2,1) =  [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2,2) = [...
                1;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2,3) =  [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3,1) = [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3,2) =  [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3,3) = [...
                0;
                0;
                0;
                0;
                0];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System('DecimationFactor',decch(1:2),'NumberOfChannels',decch(3:end),'PolyPhaseOrder',ord,'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 0 2
        function testConstructorWithDec22Ch5Ord20(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 2 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                1 -1  1 -1 ;
                1  1 -1 -1 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch5Ord22(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 0 4
        function testConstructorWithDec22Ch5Ord04(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 4 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            coefExpctd(:,:,1,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 0 4
        function testConstructorWithDec22Ch5Ord04Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 4 ];
            ang = 2*pi*rand(4,3);
            
            % Expected values
            nDecs = prod(decch(1:2));
            nChs = decch(3);
            dimExpctd = [nChs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 4 0
        function testConstructorWithDec22Ch5Ord40(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 4 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            coefExpctd(:,:,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4 0
        function testConstructorWithDec22Ord40Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 4 0 ];
            ang = 2*pi*rand(4,3);
            
            % Expected values
            nDecs = prod(decch(1:2));
            nChs = decch(3);
            dimExpctd = [nChs nDecs ord(1)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 4 4
        function testConstructorWithDec22Ch5Ord44(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 4 4 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            coefExpctd(:,:,4,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
 
        % Test dec 2 2 ch 5 order 6 6
        function testConstructorWithDec22Ch5Ord66Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 6 6 ];
            ang = 2*pi*rand(4,7);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end
        
        % Test dec 2 2 ch 9 order 2 2
        function testConstructorWithDec22Ch9Ord22Ang(testCase)
            
            % Parameters
            decch = [ 2 2 9 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(16,3);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end
        
        % Test dec 2 2 ch 9 order 4 4
        function testConstructorWithDec22Ch9Ord44Ang(testCase)
            
            % Parameters
            decch = [ 2 2 9 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(16,5);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end
        
        % Test dec 2 2 ch 11 order 4 4
        function testConstructorWithDec22Ch11Ord44Ang(testCase)
            
            % Parameters
            decch = [ 2 2 11 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(25,3);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end
        
        % Test dec 3 3 ch 11 order 2 2
        function testConstructorWithDec33Ch11Ord22Ang(testCase)
            
            % Parameters
            decch = [ 3 3 11 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(25,3);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end
        
        % Test dec 3 3 ch 11 order 4 4
        function testConstructorWithDec33Ch11Ord44Ang(testCase)
            
            % Parameters
            decch = [ 3 3 11 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(25,5);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end
        
        % Test dec 1 1 ch 5 order 2 2
        function testConstructorWithDec11Ch5Ord22Ang(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(4,3);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end
        
        % Test dec 1 1 ch 5 order 4 4
        function testConstructorWithDec11Ch5Ord44Ang(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(4,5);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end
        
        % Test dec 1 1 ch 7 order 4 4
        function testConstructorWithDec11Ch7Ord44Ang(testCase)
            
            % Parameters
            decch = [ 1 1 7 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(9,5);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end
        
        % Test dec 1 1 ch 7 order 6 6
        function testConstructorWithDec11Ch7Ord66Ang(testCase)
            
            % Parameters
            decch = [ 1 1 7 ];
            ord = [ 6 6 ];
            ang = 2*pi*rand(9,7);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),....
                'NumberOfChannels',decch(3:end),...
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
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
        end
        
        % Test for construction
        function testConstructorWithDec22Ch32Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System('DecimationFactor',decch(1:2),'NumberOfChannels',decch(3:end),'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch42Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 4 2 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System('DecimationFactor',decch(1:2),'NumberOfChannels',decch(3:end),'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch43Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 4 3 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ;
                0  0  0  0  ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System('DecimationFactor',decch(1:2),'NumberOfChannels',decch(3:end),'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch52Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 5 2 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System('DecimationFactor',decch(1:2),'NumberOfChannels',decch(3:end),'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch62Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 6 2 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System('DecimationFactor',decch(1:2),'NumberOfChannels',decch(3:end),'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch32Ord22(testCase)
            
            % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch42Ord22(testCase)
            
            % Parameters
            decch = [ 2 2 4 2 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch52Ord22(testCase)
            
            % Parameters
            decch = [ 2 2 5 2 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch53Ord22(testCase)
            
            % Parameters
            decch = [ 2 2 5 3 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ;
                0  0  0  0  ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch62Ord22(testCase)
            
            % Parameters
            decch = [ 2 2 6 2 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 4 4
        function testConstructorWithDec22Ch32Ord44(testCase)
            
            % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 4 4 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            coefExpctd(:,:,4,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end     
        
        % Test dec 2 2 order 4 4
        function testConstructorWithDec22Ch42Ord44(testCase)
            
            % Parameters
            decch = [ 2 2 4 2 ];
            ord = [ 4 4 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,1) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,1) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,2) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,2) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = 1/2 * [
                1  1  1  1 ;
                1 -1 -1  1 ;
                0  0  0  0 ;
                0  0  0  0 ;                
                -1  1 -1  1 ;
                -1 -1  1  1 ];
            
            coefExpctd(:,:,4,3) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,3) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,4) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,4) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,4) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,4) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,5) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,5) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,5) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,5) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,5) = [
                0  0  0  0 ;
                0  0  0  0 ;                
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end  
        
        % Test dec 2 2 order 4 4
        function testConstructorWithDec22Ch32Ord22Ang(testCase)
            
          % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(4,3);
            
            % Expected values
            nDec = prod(decch(1:2));
            nChs = sum(decch(3:4));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(3),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(decch(3)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 4 4
        function testConstructorWithDec22Ch32Ord44Ang(testCase)
            
            % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(4,5);
            
            % Expected values
            nDecs = prod(decch(1:2));
            nChs = sum(decch(3:4));
            dimExpctd = [nChs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(3),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(decch(3)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 4 4
        function testConstructorWithDec22Ch42Ord22Ang(testCase)
            
          % Parameters
            decch = [ 2 2 4 2 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(7,3);
            
            % Expected values
            nDec = prod(decch(1:2));
            nChs = sum(decch(3:4));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(3),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(decch(3)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 4 4
        function testConstructorWithDec22Ch42Ord44Ang(testCase)
            
            % Parameters
            decch = [ 2 2 4 2 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(7,5);
            
            % Expected values
            nDecs = prod(decch(1:2));
            nChs = sum(decch(3:4));
            dimExpctd = [nChs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(3),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            coefOdd = coefActual(decch(3)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch5Ord22AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(4,3);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch5Ord44AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(4,5);
            
            % Expected values
            nChs = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch42Ord44AngNoDcLeakage(testCase)
            
            % Parameters
            decch = [ 2 2 4 2 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(7,5);
            
            % Expected values
            nChs = sum(decch(3:4));
            nDec = prod(decch(1:2));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
        function testParameterMatrixSet(testCase)
            
            % Preparation
            mstab = [ 3 3 ; 2 2 ];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixSet
            paramExpctd = ParameterMatrixSet(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(3),1);
            step(paramExpctd,eye(2),2);
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end
        
        % Test for construction with order 2 2
        function testParameterMatrixSetRandAngWithDec22Ch32Ord22(testCase)
            
            % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 2 2 ];
            mstab = [ 3 3 ; 2 2 ; 3 3 ; 2 2 ; 3 3 ; 2 2 ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixSet(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
            step(paramMtxExpctd, eye(mstab(3,:)),uint32(3)); % Wx1
            step(paramMtxExpctd,-eye(mstab(4,:)),uint32(4)); % Ux1
            step(paramMtxExpctd, eye(mstab(5,:)),uint32(5)); % Wy1
            step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % Uy1
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            Wx1 = step(paramMtxActual,[],uint32(3));
            Wy1 = step(paramMtxActual,[],uint32(5));
            G = Wy1*Wx1*W0;
            coefActual = G(1,1);

            % Evaluation
            diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
            
            % Check DC-E
            release(testCase.lppufb)
            import matlab.unittest.constraints.IsLessThan
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            for iSubband = 2:sum(decch(3:4))
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end            
        end
        
        
        % Test for construction with order 2 2
        function testParameterMatrixSetRandMusWithDec22Ch32Ord22(testCase)
            
            % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 2 2 ];
            mstab = [ 3 3 ; 2 2 ; 3 3 ; 2 2 ; 3 3 ; 2 2 ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixSet(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
            step(paramMtxExpctd, eye(mstab(3,:)),uint32(3)); % Wx1
            step(paramMtxExpctd,-eye(mstab(4,:)),uint32(4)); % Ux1
            step(paramMtxExpctd, eye(mstab(5,:)),uint32(5)); % Wy1
            step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % Uy1
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            Wx1 = step(paramMtxActual,[],uint32(3));
            Wy1 = step(paramMtxActual,[],uint32(5));
            G = Wy1*Wx1*W0;
            coefActual = G(1,1);

            % Evaluation
            diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
            
            % Check DC-E
            release(testCase.lppufb)
            import matlab.unittest.constraints.IsLessThan
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            for iSubband = 2:sum(decch(3:4))
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end            
        end

                
        % Test for construction with order 2 2
        function testParameterMatrixSetRandAngMusWithDec22Ch32Ord22(testCase)
            
            % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 2 2 ];
            mstab = [ 3 3 ; 2 2 ; 3 3 ; 2 2 ; 3 3 ; 2 2 ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            paramMtxExpctd = ParameterMatrixSet(...
                'MatrixSizeTable',mstab);
            step(paramMtxExpctd, eye(mstab(1,:)),uint32(1)); % W0
            step(paramMtxExpctd, eye(mstab(2,:)),uint32(2)); % U0
            step(paramMtxExpctd, eye(mstab(3,:)),uint32(3)); % Wx1
            step(paramMtxExpctd,-eye(mstab(4,:)),uint32(4)); % Ux1
            step(paramMtxExpctd, eye(mstab(5,:)),uint32(5)); % Wy1
            step(paramMtxExpctd,-eye(mstab(6,:)),uint32(6)); % Uy1
            coefExpctd = get(paramMtxExpctd,'Coefficients');
            
            % Instantiation of target class
            import saivdr.dictionary.nsolt.*
            testCase.lppufb = OvsdLpPuFb2dTypeIIVm1System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            Wx1 = step(paramMtxActual,[],uint32(3));
            Wy1 = step(paramMtxActual,[],uint32(5));
            G = Wy1*Wx1*W0;
            coefActual = G(1,1);

            % Evaluation
            diff = max(abs(coefExpctd-coefActual)./abs(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',diff));
            
            % Check DC-E
            release(testCase.lppufb)
            import matlab.unittest.constraints.IsLessThan
            set(testCase.lppufb,'OutputMode','AnalysisFilterAt');
            for iSubband = 2:sum(decch(3:4))
                H = step(testCase.lppufb,[],[],iSubband);
                dc = abs(sum(H(:)));
                testCase.verifyThat(dc,IsLessThan(1e-14),sprintf('%g',dc));
            end            
        end

        
        
    end
    
end
