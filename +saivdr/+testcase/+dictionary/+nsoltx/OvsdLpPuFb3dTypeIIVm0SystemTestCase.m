classdef OvsdLpPuFb3dTypeIIVm0SystemTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB3dTYPEIIVM0SYSTEMTESTCASE Test case for OvsdLpPuFb3dTypeIIVm0System
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
                testCase.matrixE0(1:4,:) ;
                zeros(1,8);
                testCase.matrixE0(5:end,:) 
            ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System();
            
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
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System();
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
        function testConstructorWithDec222Ch9Ord000(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1,1) =  [ 
                testCase.matrixE0(1:4,:) ;
                zeros(1,8);
                testCase.matrixE0(5:end,:) 
            ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 0 0 2
        function testConstructorWithDec222Ch9Ord002Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 0 0 2 ];
            ang = 2*pi*rand(16,2);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction with order 2 0
        function testConstructorWithDec222Ch9Ord020Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 0 2 0 ];
            ang = 2*pi*rand(16,2);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction with order 2 2
        function testConstructorWithDec222Ch9Ord200Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 2 0 0 ];
            ang = 2*pi*rand(16,2);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2 2
        function testConstructorWithDec222Ord222(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(16,4);
            
            % Expected values
            nDec = prod(decch(1:3));
            nChs = decch(4);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction with order 2 0
        function testConstructorWithDec333Ord000(testCase)
            
            % Parameters
            dec = [ 3 3 3 ];
            ord = [ 0 0 0 ];
            ang = 2*pi*rand(169,1);
            
            % Expected values
            nDec = prod(dec(1:3));
            nChs = nDec;
            dimExpctd = [nChs nDec ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction with order 2 2 2
        function testConstructorWithDec222Ord222Ang(testCase)
            
            % Parameters
            dec = [ 2 2 2 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(16,4);
            
            % Expected values
            nDec = prod(dec(1:3));
            nChs = nDec + (mod(nDec,2)==0);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
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
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
        function testConstructorWithDec222Ch13Ord000(testCase)
            
            % Parameters
            decch = [ 2 2 2 13 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            dimExpctd = [13 8];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
        function testConstructorWithDec222Ch13Ord000Ang0(testCase)
            
            % Parameters
            decch = [ 2 2 2 13 ];
            ord = [ 0 0 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1,1) = [
                testCase.matrixE0(1:4,:);
                zeros(3,8);
                testCase.matrixE0(5:8,:);
                zeros(2,8)
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec222Ch13Ord000Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 13 ];
            ord = [ 0 0 0 ];
            angW = zeros(21,1);
            angU = 2*pi*rand(15,1);
            
            % Expected values
            import saivdr.dictionary.utility.*
            omgsW = OrthonormalMatrixGenerationSystem();
            omgsU = OrthonormalMatrixGenerationSystem();
            matrixW0 = step(omgsW,angW,1);
            matrixU0 = step(omgsU,angU,1);
            coefExpctd(:,:,1,1) = ...
                blkdiag(matrixW0, matrixU0) * ...
                [
                    testCase.matrixE0(1:4,:);
                    zeros(3,8);
                    testCase.matrixE0(5:8,:);
                    zeros(2,8)
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[angW;angU],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-12,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec222Ch9Ord000Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 0 0 0 ];
            ang = 2*pi*rand(16,1);
            
            % Expected values
            dimExpctd = [9 8];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
        function testConstructorWithDec222Ch7Ord000Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 11 ];
            ord = [ 0 0 0 ];
            ang = 2*pi*rand(25,1);
            
            % Expected values
            dimExpctd = [11 8];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
        function testConstructorWithDec222Ch54Ord000Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 5 4 ];
            ord = [ 0 0 0 ];
            ang = 2*pi*rand(16,1);
            
            % Expected values
            dimExpctd = [9 8];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
        function testConstructorWithDec333Ch29Ord000Ang(testCase)
            
            % Parameters
            decch = [ 3 3 3 29 ];
            ord = [ 0 0 0 ];
            ang = 2*pi*rand(196,1);
            
            % Expected values
            dimExpctd = [29 27];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec111Ch5Ord000Ang0(testCase)
            
            % Parameters
            decch = [ 1 1 1 5 ];
            ord = [ 0 0 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1,1) = [
                1 ;
                0 ;
                0 ;
                0 ;
                0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec111Ch5Ord000AngPi3(testCase)
            
            % Parameters
            decch = [ 1 1 1 5 ];
            ord = [ 0 0 0 ];
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
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
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
            decch = [ 2 2 2 9 ];
            ord = [ 0 0 0 ];
            sizeInvalid = [2 2];
            ang = 2*pi*rand(sizeInvalid);
            
            % Expected value
            sizeExpctd = [16 1];
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Size of angles must be [ %d %d ]',...
                sizeExpctd(1), sizeExpctd(2));
            
            % Instantiation of target class
            try
                import saivdr.dictionary.nsoltx.*
                testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
            ang = zeros(16,1);
            mus = [ 1 1 1 1 1 -1 -1 -1 -1 ].';
            
            % Expected values
            coefExpctd(:,:,1,1,1) = [...
                testCase.matrixE0(1:4,:);
                zeros(1,8);
               -testCase.matrixE0(5:8,:)
               ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2 2
        function testConstructorWithDec222Ch9Ord222Ang0(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 2 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(9,8,3,3,3);
            coefExpctd(:,:,2,2,2) = [
                testCase.matrixE0(1:4,:);
                zeros(1,8);
                testCase.matrixE0(5:8,:)
               ];                
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4 4 4
        function testConstructorWithDec222Ch9Ord444Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 9 ];
            ord = [ 4 4 4 ];
            ang = 2*pi*rand(16,7);
            
            % Expected values
            nDec = prod(decch(1:3));
            dimExpctd = [decch(4) nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test dec 3 3 order 2 2
        function testConstructorWithDec333Ch27Ord222Ang0(testCase)
            
            % Parameters
            decch = [ 3 3 3 27 ];
            ord = [ 2 2 2 ];
            ang = 0;
            
            % Expected values
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',[ 0 0 0 ] );
            
            % Actual values
            coefExpctd = zeros(27,27,3,3,3);
            matrixE0_ = step(testCase.lppufb,[],[]);            
            coefExpctd(:,:,2,2,2) = [
                matrixE0_(1:14,:);
                matrixE0_(15:27,:) 
               ]; 
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));            
            
        end

        % Test dec 2 2 2 order 2 2 2
        function testConstructorWithDec222Ch11Ord222Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 11 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(25,4);
            
            % Expected values
            nDec = prod(decch(1:3));
            nChs = decch(4);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
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
            ang = 2*pi*rand(169,7);
            
            % Expected values
            nDec = prod(decch(1:3));
            nChs = decch(4);
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
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
            angPre = pi/4*ones(16,1);
            angPst = zeros(16,1);
            
            % Expected values
            coefExpctd(:,:,1,1,1) = [...
                testCase.matrixE0(1:4,:);
                zeros(1,8);
                testCase.matrixE0(5:8,:)
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
            ord = [ 0 0 0 ];
            ang = zeros(16,1);
            musPre = [ 1 -1  1 -1 1 1 -1 1 -1 ].';
            musPst = 1;
            
            % Expected values
            coefExpctd(:,:,1,1,1) = [
                testCase.matrixE0(1:4,:);
                zeros(1,8);
                testCase.matrixE0(5:8,:)
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
                1  1 ;
                1  1
                ]; 
            anFiltExpctd1(:,:,2) = 1/(2*sqrt(2)) * [
                1  1 ; 
                1  1
                ];             
            anFiltExpctd2(:,:,1) = 1/(2*sqrt(2)) * [
                1 -1 ;
                1 -1
                ];                
            anFiltExpctd2(:,:,2) = 1/(2*sqrt(2)) * [
               -1  1 ;  
               -1  1 
                ];                            
            anFiltExpctd3(:,:,1) = 1/(2*sqrt(2)) * [
                1 -1
               -1  1
                ];                
            anFiltExpctd3(:,:,2) = 1/(2*sqrt(2)) * [
                1 -1
               -1  1
                ];                            
            anFiltExpctd4(:,:,1) = 1/(2*sqrt(2)) * [
                1  1
               -1 -1
                ];                
            anFiltExpctd4(:,:,2) = 1/(2*sqrt(2)) * [
               -1 -1
                1  1
                ];   
            anFiltExpctd5 = zeros(2,2,2);
            anFiltExpctd6(:,:,1) = 1/(2*sqrt(2)) * [
               -1 -1
               -1 -1
                ];                
            anFiltExpctd6(:,:,2) = 1/(2*sqrt(2)) * [
                1  1
                1  1 
                ];                            
            anFiltExpctd7(:,:,1) = 1/(2*sqrt(2)) * [
               -1  1
               -1  1
                ];                
            anFiltExpctd7(:,:,2) = 1/(2*sqrt(2)) * [
               -1  1
               -1  1 
                ];                            
            anFiltExpctd8(:,:,1) = 1/(2*sqrt(2)) * [
               -1  1
                1 -1
                ];                
            anFiltExpctd8(:,:,2) = 1/(2*sqrt(2)) * [
                1 -1
               -1  1 
                ];                
            anFiltExpctd9(:,:,1) = 1/(2*sqrt(2)) * [
               -1 -1
                1  1
                ];
            anFiltExpctd9(:,:,2) = 1/(2*sqrt(2)) * [
               -1 -1
                1  1
                ];            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
                1  1 ;
                1  1
                ]; 
            anFiltExpctd1(:,:,2) = 1/(2*sqrt(2)) * [
                1  1 ; 
                1  1
                ];             
            anFiltExpctd2(:,:,1) = 1/(2*sqrt(2)) * [
                1 -1 ;
                1 -1
                ];                
            anFiltExpctd2(:,:,2) = 1/(2*sqrt(2)) * [
               -1  1 ;  
               -1  1 
                ];                            
            anFiltExpctd3(:,:,1) = 1/(2*sqrt(2)) * [
                1 -1
               -1  1
                ];                
            anFiltExpctd3(:,:,2) = 1/(2*sqrt(2)) * [
                1 -1
               -1  1
                ];                            
            anFiltExpctd4(:,:,1) = 1/(2*sqrt(2)) * [
                1  1
               -1 -1
                ];                
            anFiltExpctd4(:,:,2) = 1/(2*sqrt(2)) * [
               -1 -1
                1  1
                ];   
            anFiltExpctd5 = zeros(2,2,2);
            anFiltExpctd6(:,:,1) = 1/(2*sqrt(2)) * [
               -1 -1
               -1 -1
                ];                
            anFiltExpctd6(:,:,2) = 1/(2*sqrt(2)) * [
                1  1
                1  1 
                ];                            
            anFiltExpctd7(:,:,1) = 1/(2*sqrt(2)) * [
               -1  1
               -1  1
                ];                
            anFiltExpctd7(:,:,2) = 1/(2*sqrt(2)) * [
               -1  1
               -1  1 
                ];                            
            anFiltExpctd8(:,:,1) = 1/(2*sqrt(2)) * [
               -1  1
                1 -1
                ];                
            anFiltExpctd8(:,:,2) = 1/(2*sqrt(2)) * [
                1 -1
               -1  1 
                ];                
            anFiltExpctd9(:,:,1) = 1/(2*sqrt(2)) * [
               -1 -1
                1  1
                ];
            anFiltExpctd9(:,:,2) = 1/(2*sqrt(2)) * [
               -1 -1
                1  1
                ];            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
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
                testCase.matrixE0(1:4,:)
                zeros(1,8)
                testCase.matrixE0(5:8,:)
               ];

            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test dec 1 1 1 ch 5 order 0 2
        function testConstructorWithDec11Ch5Ord002(testCase)
            
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
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test dec 1 1 1 ch 5 order 0 2 2 
        function testConstructorWithDec11Ch4Ord22(testCase)
            
            % Parameters
            decch = [ 1 1 1 5 ];
            ord = [ 0 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(5,1,1,3,3);        
            coefExpctd(:,:,1,2,2) = [...
                1;
                0;
                0;
                0;
                0];

            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test dec 2 2 2 order 4 4 4
        function testConstructorWithDec222Ch64Ord444(testCase)
            
            % Parameters
            decch = [ 2 2 2 6 4 ];
            ord = [ 4 4 4 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(10,8,5,5,5);
            coefExpctd(:,:,3,3,3) =  [
                testCase.matrixE0(1:4,:)
                zeros(2,8);
                testCase.matrixE0(5:8,:)
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end  
        
        % Test dec 2 2 2 order 2 2 2
        function testConstructorWithDec222Ch74Ord222Ang(testCase)
            
          % Parameters
            decch = [ 2 2 2 7 4 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(27,4);
            
            % Expected values
            nDec = prod(decch(1:3));
            nChs = sum(decch(4:end));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(4),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(decch(4)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 2 order 4 4 4
        function testConstructorWithDec222Ch64Ord444Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 6 4 ];
            ord = [ 4 4 4 ];
            ang = 2*pi*rand(21,7);
            
            % Expected values
            nDec = prod(decch(1:3));
            nChs = sum(decch(4:end));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(4),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(decch(4)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test dec 2 2 2 order 2 2 2
        function testConstructorWithDec222Ch64Ord222Ang(testCase)
            
          % Parameters
            decch = [ 2 2 2 6 4 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(21,4);
            
            % Expected values
            nDec = prod(decch(1:3));
            nChs = sum(decch(4:end));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(4),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),sprintf('%g',coefDist));
            coefOdd = coefActual(decch(4)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        %}
  
        % Test for ParameterMatrixSet
        function testParameterMatrixContainer(testCase)
            
            % Preparation
            mstab = [ 5 5 ; 4 4 ];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixContainer
            paramExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(5),1);
            step(paramExpctd,eye(4),2);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end

  
        function testStepOrd222Ch54Rand(testCase)

            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            omgW = OrthonormalMatrixGenerationSystem();
            omgU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            ord  = [ 2 2 2 ];
            nch  = [ 5 4 ];
            npmU = 10;
            npmL = 6;
            angs = rand(npmU+npmL,(2+sum(ord))/2);
            mus  = ones(sum(nch),(2+sum(ord))/2);
            nchn = min(nch);
            nchx = max(nch);
            In   = eye(nchn);
            Ix   = eye(nchx);
            Znx  = zeros(nchn,nchx);
            Zxn  = zeros(nchx,nchn);
            Zn   = zeros(nchn);
            Zx   = zeros(nchx);
            %
            Dzo = zeros(9,9,1,1,2);
            Dzo(:,:,1,1,1) = [ In Znx ; Zxn Zx ];
            Dzo(:,:,1,1,2) = [ Zn Znx ; Zxn Ix ];
            Dze = zeros(9,9,1,1,2);
            Dze(:,:,1,1,1) = [ Ix Zxn ; Znx Zn ];
            Dze(:,:,1,1,2) = [ Zx Zxn ; Znx In ];            
            %
            Dxo = zeros(9,9,1,2,1);
            Dxo(:,:,1,1,1) = [ In Znx ; Zxn Zx ];
            Dxo(:,:,1,2,1) = [ Zn Znx ; Zxn Ix ];
            Dxe = zeros(9,9,1,2,1);
            Dxe(:,:,1,1,1) = [ Ix Zxn ; Znx Zn ];
            Dxe(:,:,1,2,1) = [ Zx Zxn ; Znx In ];                        
            %
            Dyo = zeros(9,9,2,1,1);
            Dyo(:,:,1,1,1) = [ In Znx ; Zxn Zx ];
            Dyo(:,:,2,1,1) = [ Zn Znx ; Zxn Ix ];
            Dye = zeros(9,9,2,1,1);
            Dye(:,:,1,1,1) = [ Ix Zxn ; Znx Zn ];
            Dye(:,:,2,1,1) = [ Zx Zxn ; Znx In ];
            %
            W0  = step(omgW,angs(1:npmU,1),mus(1:nchx,1));
            U0  = step(omgU,angs(npmU+1:end,1),mus(nchx+1:end,1));
            Wz1 = step(omgW,angs(1:npmU,2),mus(1:nchx,2));
            Uz1 = step(omgU,angs(npmU+1:end,2),mus(nchx+1:end,2));
            Wx1 = step(omgW,angs(1:npmU,3),mus(1:nchx,3));
            Ux1 = step(omgU,angs(npmU+1:end,3),mus(nchx+1:end,3));
            Wy1 = step(omgW,angs(1:npmU,4),mus(1:nchx,4));
            Uy1 = step(omgU,angs(npmU+1:end,4),mus(nchx+1:end,4));
            %
            Znd  = zeros(nchn,(nchx-nchn));
            Id  = eye(nchx-nchn);
            B  = PolyPhaseMatrix3d([
                In    Znd     In ; 
                Znd.' sqrt(2)*Id Znd.'
                In    Znd     -In 
                ]/sqrt(2));
            Qzo = B*PolyPhaseMatrix3d(Dzo)*B;                        
            Qze = B*PolyPhaseMatrix3d(Dze)*B;
            Qxo = B*PolyPhaseMatrix3d(Dxo)*B;            
            Qxe = B*PolyPhaseMatrix3d(Dxe)*B;
            Qyo = B*PolyPhaseMatrix3d(Dyo)*B;                
            Qye = B*PolyPhaseMatrix3d(Dye)*B;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',[ 2 2 2 ],...
                'NumberOfChannels',[ 5 4 ],...
                'PolyPhaseOrder',ord,...
                'OutputMode','Coefficients');            
            set(testCase.lppufb,'Angles',angs);
            set(testCase.lppufb,'Mus',mus);

            % Expected values
            import saivdr.dictionary.utility.PolyPhaseMatrix3d
            E0 = testCase.matrixE0;
            R0 = blkdiag(W0,U0)*[In Zn; Znd.' Znd.' ;Zn In];
            Rz1 = blkdiag(Ix,Uz1);
            Rz2 = blkdiag(Wz1,In);
            Rx1 = blkdiag(Ix,Ux1);
            Rx2 = blkdiag(Wx1,In);
            Ry1 = blkdiag(Ix,Uy1);
            Ry2 = blkdiag(Wy1,In);
            E = Ry2*Qye*Ry1*Qyo*Rx2*Qxe*Rx1*Qxo*Rz2*Qze*Rz1*Qzo*R0*E0;
            
            % Actual values
            ordExpctd = ord;
            cfsExpctd = E.Coefficients;
                                    
            ordActual = get(testCase.lppufb,'PolyPhaseOrder');
            cfsActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8);
            
        end            
        % Test dec 2 2 2 order 4 4 4
        function testConstructorWithDec222Ch46Ord444(testCase)
            
            % Parameters
            decch = [ 2 2 2 4 6 ];
            ord = [ 4 4 4 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(10,8,5,5,5);
            coefExpctd(:,:,3,3,3) =  [
                testCase.matrixE0(1:4,:)
                testCase.matrixE0(5:8,:)
                zeros(2,8);
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end  
        
        % Test dec 2 2 2 order 2 2 2
        function testConstructorWithDec222Ch47Ord222Ang(testCase)
            
          % Parameters
            decch = [ 2 2 2 4 7 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(27,4);
            
            % Expected values
            nDec = prod(decch(1:3));
            nChs = sum(decch(4:end));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(4),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(decch(4)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 2 order 4 4 4
        function testConstructorWithDec222Ch46Ord444Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 4 6 ];
            ord = [ 4 4 4 ];
            ang = 2*pi*rand(21,7);
            
            % Expected values
            nDec = prod(decch(1:3));
            nChs = sum(decch(4:end));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(4),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(decch(4)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test dec 2 2 2 order 2 2 2
        function testConstructorWithDec222Ch46Ord222Ang(testCase)
            
          % Parameters
            decch = [ 2 2 2 4 6 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(21,4);
            
            % Expected values
            nDec = prod(decch(1:3));
            nChs = sum(decch(4:end));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan
            coefEvn = coefActual(1:decch(4),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),sprintf('%g',coefDist));
            coefOdd = coefActual(decch(4)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = norm(coefDiff(:))/sqrt(numel(coefDiff));
            testCase.verifyThat(coefDist,IsLessThan(1e-15),sprintf('%g',coefDist));
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
    
        % Test for ParameterMatrixSet
        function testParameterMatrixSetCh45(testCase)
            
            % Preparation
            chs = [ 4 5 ];
            mstab = [ 4 4 ; 5 5 ];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixContainer
            paramExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(4),1);
            step(paramExpctd,eye(5),2);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'OutputMode','ParameterMatrixSet',...
                'NumberOfChannels',chs);
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end

        function testStepOrd222Ch45Rand(testCase)

            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            omgW = OrthonormalMatrixGenerationSystem();
            omgU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            ord  = [ 2 2 2 ];
            nch  = [ 4 5 ];
            npmU = 6;
            npmL = 10;
            angs = rand(npmU+npmL,(2+sum(ord))/2);
            mus  = ones(sum(nch),(2+sum(ord))/2);
            nchn = min(nch);
            nchx = max(nch);
            In   = eye(nchn);
            Ix   = eye(nchx);
            Znx  = zeros(nchn,nchx);
            Zxn  = zeros(nchx,nchn);
            Zn   = zeros(nchn);
            Zx   = zeros(nchx);
            %
            Dzo = zeros(9,9,1,1,2);
            Dzo(:,:,1,1,1) = [ In Znx ; Zxn Zx ];
            Dzo(:,:,1,1,2) = [ Zn Znx ; Zxn Ix ];
            Dze = zeros(9,9,1,1,2);
            Dze(:,:,1,1,1) = [ Ix Zxn ; Znx Zn ];
            Dze(:,:,1,1,2) = [ Zx Zxn ; Znx In ];            
            %
            Dxo = zeros(9,9,1,2,1);
            Dxo(:,:,1,1,1) = [ In Znx ; Zxn Zx ];
            Dxo(:,:,1,2,1) = [ Zn Znx ; Zxn Ix ];
            Dxe = zeros(9,9,1,2,1);
            Dxe(:,:,1,1,1) = [ Ix Zxn ; Znx Zn ];
            Dxe(:,:,1,2,1) = [ Zx Zxn ; Znx In ];                        
            %
            Dyo = zeros(9,9,2,1,1);
            Dyo(:,:,1,1,1) = [ In Znx ; Zxn Zx ];
            Dyo(:,:,2,1,1) = [ Zn Znx ; Zxn Ix ];
            Dye = zeros(9,9,2,1,1);
            Dye(:,:,1,1,1) = [ Ix Zxn ; Znx Zn ];
            Dye(:,:,2,1,1) = [ Zx Zxn ; Znx In ];
            %
            W0  = step(omgW,angs(1:npmU,1),mus(1:nchn,1));
            U0  = step(omgU,angs(npmU+1:end,1),mus(nchn+1:end,1));
            Wz1 = step(omgW,angs(1:npmU,2),mus(1:nchn,2));
            Uz1 = step(omgU,angs(npmU+1:end,2),mus(nchn+1:end,2));
            Wx1 = step(omgW,angs(1:npmU,3),mus(1:nchn,3));
            Ux1 = step(omgU,angs(npmU+1:end,3),mus(nchn+1:end,3));
            Wy1 = step(omgW,angs(1:npmU,4),mus(1:nchn,4));
            Uy1 = step(omgU,angs(npmU+1:end,4),mus(nchn+1:end,4));
            %
            Znd  = zeros(nchn,(nchx-nchn));
            Id  = eye(nchx-nchn);
            B  = PolyPhaseMatrix3d([
                In    Znd     In ; 
                Znd.' sqrt(2)*Id Znd.'
                In    Znd     -In 
                ]/sqrt(2));
            Qzo = B*PolyPhaseMatrix3d(Dzo)*B;                        
            Qze = B*PolyPhaseMatrix3d(Dze)*B;
            Qxo = B*PolyPhaseMatrix3d(Dxo)*B;            
            Qxe = B*PolyPhaseMatrix3d(Dxe)*B;
            Qyo = B*PolyPhaseMatrix3d(Dyo)*B;                
            Qye = B*PolyPhaseMatrix3d(Dye)*B;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIIVm0System(...
                'DecimationFactor',[ 2 2 2 ],...
                'NumberOfChannels',[ 4 5 ],...
                'PolyPhaseOrder',ord,...
                'OutputMode','Coefficients');            
            set(testCase.lppufb,'Angles',angs);
            set(testCase.lppufb,'Mus',mus);

            % Expected values
            import saivdr.dictionary.utility.PolyPhaseMatrix3d
            E0 = testCase.matrixE0;
            R0 = blkdiag(W0,U0)*[In Zn; Zn In; Znd.' Znd.' ];
            Rz1 = blkdiag(Wz1,Ix);
            Rz2 = blkdiag(In,Uz1);
            Rx1 = blkdiag(Wx1,Ix);
            Rx2 = blkdiag(In,Ux1);
            Ry1 = blkdiag(Wy1,Ix);
            Ry2 = blkdiag(In,Uy1);
            E = Ry2*Qye*Ry1*Qyo*Rx2*Qxe*Rx1*Qxo*Rz2*Qze*Rz1*Qzo*R0*E0;
            
            % Actual values
            ordExpctd = ord;
            cfsExpctd = E.Coefficients;
                                    
            ordActual = get(testCase.lppufb,'PolyPhaseOrder');
            cfsActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8);
            
        end            
        
    end
    
end
