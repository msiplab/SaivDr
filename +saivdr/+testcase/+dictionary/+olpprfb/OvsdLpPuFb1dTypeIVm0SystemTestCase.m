classdef OvsdLpPuFb1dTypeIVm0SystemTestCase < matlab.unittest.TestCase
    %OvsdLpPuFb1dTypeIVm0SystemTESTCASE Test case for OvsdLpPuFb1dTypeIVm0System
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
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000 ;
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000 ;
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188 ;
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System();
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for default construction
        function testConstructorWithDeepCopy(testCase)
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System();
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
        function testConstructorWithOrd0(testCase)
            
            % Parameters
            dec = 4;
            ord = 0;
            
            % Expected values
            coefExpctd = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000 ;
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000 ;
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188 ;
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
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
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        function testConstructorWithDec4Ch4Ord0(testCase)
            
            % Parameters
            decch = [ 4 4 ];
            ord   =  0;
            
            % Expected values
            coefExpctd = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000 ;
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000 ;
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188 ;
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 0
        function testConstructorWithDec4Ch6Ord0(testCase)
            
            % Parameters
            decch = [ 4 6 ];
            ord = 0;
            ang = 2*pi*rand(3,2);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        
        % Test for construction with order 2
        function testConstructorWithDec4Ch6Ord2(testCase)
            
            % Parameters
            decch = [ 4 6 ];
            ord   = 2;
            ang   = 2*pi*rand(3,4);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        
        % Test for construction with order 2
        function testConstructorWithDec4Ch8Ord20(testCase)
            
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
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        
        % Test for construction with order 0 2
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
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        
        % Test for construction
        function testConstructorWithDec4Ch8Ord0(testCase)
            
            % Parameters
            decch = [ 4 8 ];
            ord   = 0;
            
            % Expected values
            dimExpctd = [8 4];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        function testConstructorWithOrd0Ang(testCase)
            
            % Parameters
            decch = [ 4 4 ];
            ord   = 0;
            ang   = 0;
            
            % Expected values
            coefExpctd = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000 ;
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000 ;
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188 ;
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithAng0Pi4(testCase)
            
            % Parameters
            decch = [ 4 4 ];
            ord   = 0;
            ang   = [ 0 pi/4 ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            omgs = OrthonormalMatrixGenerationSystem();
            matrixW0 = step(omgs,ang(1),1);
            matrixU0 = step(omgs,ang(2),1);
            coefExpctd = ...
                blkdiag(matrixW0, matrixU0) * ...
                [ 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000 ;
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000 ;
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188 ;
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
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
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        function testConstructorWithDec4Ch8Ord0Ang(testCase)
            
            % Parameters
            decch = [ 4 8 ];
            ord = 0;
            ang = 2*pi*rand(6,2);
            
            % Expected values
            dimExpctd = [8 4];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        function testConstructorWithDec4Ch6Ord0Ang(testCase)
            
            % Parameters
            decch = [ 4 6 ];
            ord = 0;
            ang = 2*pi*rand(3,2);
            
            % Expected values
            dimExpctd = [6 4];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        function testConstructorWithDec1Ch4Ord0(testCase)
            
            % Parameters
            decch = [ 1 4 ];
            ord = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [...
                1;
                0;
                0;
                0];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec1Ch4Ord0Ang(testCase)
            
            % Parameters
            decch = [ 1 4 ];
            ord = 0;
            ang = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1) = [
                1 ;
                0 ;
                0 ;
                0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec1Ch4Ord0Ang0Pi4(testCase)
            
            % Parameters
            decch = [ 1 4 ];
            ord = 0;
            ang = [ 0 pi/4 ];
            
            % Expected values
            import saivdr.dictionary.utility.*
            omgs = OrthonormalMatrixGenerationSystem();
            matrixW0 = step(omgs,ang(1),1);
            matrixU0 = step(omgs,ang(2),1);
            coefExpctd(:,:,1) = ...
                blkdiag(matrixW0, matrixU0) * ...
                [ 1 0 0 0 ].';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        function testConstructorWithInvalidArguments(testCase)
            
            % Invalid input
            dec = 4;
            ord = 0;
            sizeInvalid = [2 2];
            ang = 2*pi*rand(sizeInvalid);
            
            % Expected value
            sizeExpctd = [1 2];
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Size of angles must be [ %d %d ]',...
                sizeExpctd(1), sizeExpctd(2));
            
            % Instantiation of target class
            try
                import saivdr.dictionary.olpprfb.*
                OvsdLpPuFb1dTypeIVm0System(...
                    'DecimationFactor',dec,...
                    'PolyPhaseOrder',ord,...
                    'Angles',ang);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
        end
        
        function testConstructorWithOddChannels(testCase)
            
            % Invalid input
            decch = [ 4 5 ];
            ord = 0;
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = '#Channels must be even.';
            
            % Instantiation of target class
            try
                import saivdr.dictionary.olpprfb.*
                OvsdLpPuFb1dTypeIVm0System(...
                    'DecimationFactor',decch(1),...
                    'NumberOfChannels',decch(2),...
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
        
        function testConstructorWithUnEqualPsPa(testCase)
            
            % Invalid input
            decch = [ 6 2 4 ];
            ord = 0;
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = 'ps and pa must be the same as each other.';
            
            % Instantiation of target class
            try
                import saivdr.dictionary.olpprfb.*
                OvsdLpPuFb1dTypeIVm0System(...
                    'DecimationFactor',decch(1),...
                    'NumberOfChannels',decch(2:end),...
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
        
        % Test for construction
        function testConstructorWithMusPosNeg(testCase)
            
            % Parameters
            dec = 4;
            ord = 0;
            ang = [ 0 0 ];
            mus = [ 1 1 ; -1 -1 ];
            
            % Expected values
            coefExpctd(:,:,1) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000 ;
                -0.500000000000000   0.500000000000000   0.500000000000000  -0.500000000000000 ;
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188 ;
                -0.270598050073099   0.653281482438188  -0.653281482438188   0.270598050073099 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Angles',ang,...
                'Mus',mus);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithOrd2(testCase)
            
            % Parameters
            dec = 4;
            ord = 2;
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,2) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000 ;
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000 ;
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188 ;
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 ];
            coefExpctd(:,:,3) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4 4
        function testConstructorWithOrd4(testCase)
            
            % Parameters
            dec = 4;
            ord = 4;
            ang = 2*pi*rand(1,6);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 5
        function testConstructorWithOrd5(testCase)
            
            % Parameters
            dec = 4;
            ord = 5;
            ang = 2*pi*rand(1,7);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 8 order 2
        function testConstructorWithDec8Ord2(testCase)
            
            % Parameters
            dec = 8;
            ord = 2;
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = zeros(dec);
            coefExpctd(:,:,2) = [
                0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274
                0.461939766255643   0.191341716182545  -0.191341716182545  -0.461939766255643  -0.461939766255643  -0.191341716182545   0.191341716182545   0.461939766255643
                0.353553390593274  -0.353553390593274  -0.353553390593274   0.353553390593274   0.353553390593274  -0.353553390593273  -0.353553390593274   0.353553390593273
                0.191341716182545  -0.461939766255643   0.461939766255643  -0.191341716182545  -0.191341716182545   0.461939766255644  -0.461939766255644   0.191341716182543
                0.490392640201615   0.415734806151273   0.277785116509801   0.097545161008064  -0.097545161008064  -0.277785116509801  -0.415734806151273  -0.490392640201615
                0.415734806151273  -0.097545161008064  -0.490392640201615  -0.277785116509801   0.277785116509801   0.490392640201615   0.097545161008064  -0.415734806151272
                0.277785116509801  -0.490392640201615   0.097545161008064   0.415734806151273  -0.415734806151273  -0.097545161008065   0.490392640201615  -0.277785116509801
                0.097545161008064  -0.277785116509801   0.415734806151273  -0.490392640201615   0.490392640201615  -0.415734806151272   0.277785116509802  -0.097545161008063
                ];
            coefExpctd(:,:,3) = zeros(dec);
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-13,sprintf('%g',coefDist));
            
        end
        
        % Test dec 8 order 2
        function testConstructorWithDec8Ord2Ang(testCase)
            
            % Parameters
            dec = 8;
            ord = 2;
            ang = 2*pi*rand(6,4);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test: dec 8 order 4
        function testConstructorWithDec8Ord4Ang(testCase)
            
            % Parameters
            dec = 8;
            ord = 4;
            ang = 2*pi*rand(6,6);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for angle setting
        function testSetAngles(testCase)
            
            % Parameters
            dec = 4;
            ord = 0;
            angPre = [ pi/4 pi/4 ];
            angPst = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
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
        
        % Test for angle setting
        function testSetMus(testCase)
            
            % Parameters
            dec = 4;
            ord = 0;
            ang = [ 0 0 ];
            musPre = [ 1 -1 ; 1 -1 ];
            musPst = 1;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
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
        
        % Test
        function testAnalysisFilterAt(testCase)
            
            % Expected value
            anFiltExpctd1 = [ 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000 ];
            anFiltExpctd2 = [ 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000 ];
            anFiltExpctd3 = [ 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188 ];
            anFiltExpctd4 = [ 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'OutputMode','AnalysisFilterAt');
            
            % Actual values
            anFiltActual1 = step(testCase.lppufb,[],[],1);
            anFiltActual2 = step(testCase.lppufb,[],[],2);
            anFiltActual3 = step(testCase.lppufb,[],[],3);
            anFiltActual4 = step(testCase.lppufb,[],[],4);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            dist = norm(anFiltExpctd1(:)-anFiltActual1(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd2(:)-anFiltActual2(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd3(:)-anFiltActual3(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd4(:)-anFiltActual4(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            
        end
        
        function testAnalysisFilters(testCase)
            
            % Expected value
            anFiltExpctd1 = [ 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000 ].';
            anFiltExpctd2 = [ 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000 ].';
            anFiltExpctd3 = [ 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188 ].';
            anFiltExpctd4 = [ 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 ].';
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'OutputMode','AnalysisFilters');
            
            % Actual values
            anFiltsActual = step(testCase.lppufb,[],[]);
            anFiltActual1 = anFiltsActual(:,1);
            anFiltActual2 = anFiltsActual(:,2);
            anFiltActual3 = anFiltsActual(:,3);
            anFiltActual4 = anFiltsActual(:,4);
            
            % Evaluation
            dist = max(abs(anFiltExpctd1(:)-anFiltActual1(:))./abs(anFiltExpctd1(:)));
            testCase.verifyEqual(anFiltActual1,anFiltExpctd1,'RelTol',1e-14,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd2(:)-anFiltActual2(:))./abs(anFiltExpctd2(:)));
            testCase.verifyEqual(anFiltActual2,anFiltExpctd2,'RelTol',1e-14,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd3(:)-anFiltActual3(:))./abs(anFiltExpctd3(:)));
            testCase.verifyEqual(anFiltActual3,anFiltExpctd3,'RelTol',1e-14,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd4(:)-anFiltActual4(:))./abs(anFiltExpctd4(:)));
            testCase.verifyEqual(anFiltActual4,anFiltExpctd4,'RelTol',1e-14,sprintf('%g',dist));
            
        end
        
        % Test dec 4 order 2
        function testConstructorWithDec4Ord2(testCase)
            
            % Parameters
            dec = 4;
            ord = 2;
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = zeros(dec);
            coefExpctd(:,:,2) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            coefExpctd(:,:,3) = zeros(dec);
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 ch 4 order 2
        function testConstructorWithDec1Ch4Ord2(testCase)
            
            % Parameters
            decch = [ 1 4 ];
            ord = 2;
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0  ;
                0  ;
                0  ;
                0  ];
            
            coefExpctd(:,:,2) = [
                1 ;
                0 ;
                0 ;
                0 ];
            
            coefExpctd(:,:,3) = [
                0  ;
                0  ;
                0  ;
                0  ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 ch 4 order 4
        function testConstructorWithDec1Ch4Ord4(testCase)
            
            % Parameters
            decch = [ 1 4 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [...
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2) =  [...
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3) = [...
                1;
                0;
                0;
                0];
            
            coefExpctd(:,:,4) =  [...
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,5) = [...
                0;
                0;
                0;
                0];
            
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 order 2
        function testConstructorWithDec4Ch4Ord2(testCase)
            
            % Parameters
            decch = [ 4 4 ];
            ord = 2;
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099];
            
            coefExpctd(:,:,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2
        function testConstructorWithDec4Ord2Ang(testCase)
            
            % Parameters
            dec = 4;
            ord = 2;
            ang = 2*pi*rand(1,4);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 ch 4 order 2
        function testConstructorWithDec4Ch4Ord2Ang(testCase)
            
            % Parameters
            decch = [ 4 4 ];
            ord = 2;
            ang = 2*pi*rand(1,4);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        
        % Test dec 4 order 4
        function testConstructorWithDec4Ord4(testCase)
            
            % Parameters
            dec = 4;
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(dec,dec,ord+1);
            coefExpctd(:,:,3) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4
        function testConstructorWithDec4Ord4Ang(testCase)
            
            % Parameters
            dec = 4;
            ord = 4;
            ang = 2*pi*rand(1,6);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 8 order 4
        function testConstructorWithDec8Ord4(testCase)
            
            % Parameters
            dec = 8;
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(dec,dec,ord+1);
            coefExpctd(:,:,3) = [
                0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274
                0.461939766255643   0.191341716182545  -0.191341716182545  -0.461939766255643  -0.461939766255643  -0.191341716182545   0.191341716182545   0.461939766255643
                0.353553390593274  -0.353553390593274  -0.353553390593274   0.353553390593274   0.353553390593274  -0.353553390593273  -0.353553390593274   0.353553390593273
                0.191341716182545  -0.461939766255643   0.461939766255643  -0.191341716182545  -0.191341716182545   0.461939766255644  -0.461939766255644   0.191341716182543
                0.490392640201615   0.415734806151273   0.277785116509801   0.097545161008064  -0.097545161008064  -0.277785116509801  -0.415734806151273  -0.490392640201615
                0.415734806151273  -0.097545161008064  -0.490392640201615  -0.277785116509801   0.277785116509801   0.490392640201615   0.097545161008064  -0.415734806151272
                0.277785116509801  -0.490392640201615   0.097545161008064   0.415734806151273  -0.415734806151273  -0.097545161008065   0.490392640201615  -0.277785116509801
                0.097545161008064  -0.277785116509801   0.415734806151273  -0.490392640201615   0.490392640201615  -0.415734806151272   0.277785116509802  -0.097545161008063
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-13,sprintf('%g',coefDist));
            
        end
        
        % Test dec 6 order 4
        function testConstructorWithDec6Ord4Ang(testCase)
            
            % Parameters
            dec = 6;
            ord = 4;
            ang = 2*pi*rand(3,6);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 6 order 4
        function testConstructorWithDec6Ord4(testCase)
            
            % Parameters
            dec = 6;
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(6,6,5);
            coefExpctd(:,:,3) = [
                0.408248290463863   0.408248290463863   0.408248290463863   0.408248290463863   0.408248290463863   0.408248290463863
                0.500000000000000   0.000000000000000  -0.500000000000000  -0.500000000000000  -0.000000000000000   0.500000000000000
                0.288675134594813  -0.577350269189626   0.288675134594813   0.288675134594813  -0.577350269189626   0.288675134594812
                0.557677535825205   0.408248290463863   0.149429245361342  -0.149429245361342  -0.408248290463863  -0.557677535825205
                0.408248290463863  -0.408248290463863  -0.408248290463863   0.408248290463863   0.408248290463863  -0.408248290463863
                0.149429245361342  -0.408248290463863   0.557677535825205  -0.557677535825205   0.408248290463863  -0.149429245361341
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 10 order 4
        function testConstructorWithDec10Ord4Ang(testCase)
            
            % Parameters
            dec = 10;
            ord = 4;
            ang = 2*pi*rand(10,6);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 order 6
        function testConstructorWithDec4Ord6(testCase)
            
            % Parameters
            dec = 4;
            ord = 6;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(4,4,7);
            coefExpctd(:,:,4) = [
                0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000 ;
                0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000 ;
                0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188 ;
                0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 order 6
        function testConstructorWithDec4Ord6Ang(testCase)
            
            % Parameters
            dec = 4;
            ord = 6;
            ang = 2*pi*rand(1,8);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 8 order 6
        function testConstructorWithDec8Ord6(testCase)
            
            % Parameters
            dec = 8;
            ord = 6;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(8,8,7);
            coefExpctd(:,:,4) = [
                0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274
                0.461939766255643   0.191341716182545  -0.191341716182545  -0.461939766255643  -0.461939766255643  -0.191341716182545   0.191341716182545   0.461939766255643
                0.353553390593274  -0.353553390593274  -0.353553390593274   0.353553390593274   0.353553390593274  -0.353553390593273  -0.353553390593274   0.353553390593273
                0.191341716182545  -0.461939766255643   0.461939766255643  -0.191341716182545  -0.191341716182545   0.461939766255644  -0.461939766255644   0.191341716182543
                0.490392640201615   0.415734806151273   0.277785116509801   0.097545161008064  -0.097545161008064  -0.277785116509801  -0.415734806151273  -0.490392640201615
                0.415734806151273  -0.097545161008064  -0.490392640201615  -0.277785116509801   0.277785116509801   0.490392640201615   0.097545161008064  -0.415734806151272
                0.277785116509801  -0.490392640201615   0.097545161008064   0.415734806151273  -0.415734806151273  -0.097545161008065   0.490392640201615  -0.277785116509801
                0.097545161008064  -0.277785116509801   0.415734806151273  -0.490392640201615   0.490392640201615  -0.415734806151272   0.277785116509802  -0.097545161008063
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-13,sprintf('%g',coefDist));
            
        end
        
        % Test dec 8 order 6
        function testConstructorWithDec8Ord6Ang(testCase)
            
            % Parameters
            dec = 8;
            ord = 6;
            ang = 2*pi*rand(6,8);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 ch 4 order 4
        function testConstructorWithDec4Ch4Ord4Ang(testCase)
            
            % Parameters
            decch = [ 4 4 ];
            ord = 4;
            ang = 2*pi*rand(1,6);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        
        % Test dec 4 ch 6 order 2
        function testConstructorWithDec4Ch6Ord2Ang(testCase)
            
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
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        
        % Test dec 4 ch 6 order 4
        function testConstructorWithDec4Ch6Ord4Ang(testCase)
            
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
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        
        % Test dec 4 ch 8 order 2
        function testConstructorWithDec4Ch8Ord2Ang(testCase)
            
            % Parameters
            decch = [ 4 8 ];
            ord = 2;
            ang = 2*pi*rand(6,4);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        
        % Test dec 4 ch 8 order 4
        function testConstructorWithDec4Ch8Ord4Ang(testCase)
            
            % Parameters
            decch = [ 4 8 ];
            ord = 4;
            ang = 2*pi*rand(6,6);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
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
        
        % Test for construction
        function testConstructorWithOrd1(testCase)
            
            % Parameters
            dec = 4;
            ord = 1;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0.576640741219094   0.385299025036549   0.114700974963451  -0.076640741219094
                0.385299025036549  -0.576640741219094   0.076640741219094   0.114700974963451
                -0.576640741219094  -0.385299025036549  -0.114700974963451   0.076640741219094
                -0.385299025036549   0.576640741219094  -0.076640741219094  -0.114700974963451
                ];
            
            coefExpctd(:,:,2) = [
                -0.076640741219094   0.114700974963451   0.385299025036549   0.576640741219094
                0.114700974963451   0.076640741219094  -0.576640741219094   0.385299025036549
                -0.076640741219094   0.114700974963451   0.385299025036549   0.576640741219094
                0.114700974963451   0.076640741219094  -0.576640741219094   0.385299025036549
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.olpprfb.*
            testCase.lppufb = OvsdLpPuFb1dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
    end
    
end
