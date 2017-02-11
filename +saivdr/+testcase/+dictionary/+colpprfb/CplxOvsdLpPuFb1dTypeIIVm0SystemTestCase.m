classdef CplxOvsdLpPuFb1dTypeIIVm0SystemTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB1dTYPEIIVM0SYSTEMTESTCASE Test case for CplxOvsdLpPuFb1dTypeIIVm0System
    %
    % SVN identifier:
    % $Id: CplxOvsdLpPuFb1dTypeIIVm0SystemTestCase.m 240 2014-02-23 13:44:58Z sho $
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
            coefExpctd = [...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System();
            
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
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System();
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
        function testConstructorWithDec5Ord0(testCase)
            
            % Parameters
            dec = 5;
            ord = 0;
            
            % Expected values
            coefExpctd = [...
                 0.447213595499958 + 0.000000000000000i,...
                 0.447213595499958 + 0.000000000000000i,...
                 0.447213595499958 + 0.000000000000000i,...
                 0.447213595499958 + 0.000000000000000i,...
                 0.447213595499958 + 0.000000000000000i;
                 0.361803398874989 - 0.262865556059567i,...
                -0.138196601125011 - 0.425325404176020i,...
                -0.447213595499958 - 0.000000000000000i,...
                -0.138196601125011 + 0.425325404176020i,...
                 0.361803398874989 + 0.262865556059567i;
                 0.138196601125011 - 0.425325404176020i,...
                -0.361803398874989 + 0.262865556059567i,...
                 0.447213595499958 + 0.000000000000000i,...
                -0.361803398874989 - 0.262865556059567i,...
                 0.138196601125010 + 0.425325404176020i;
                -0.138196601125011 - 0.425325404176020i,...
                 0.361803398874989 + 0.262865556059567i,...
                -0.447213595499958 - 0.000000000000000i,...
                 0.361803398874989 - 0.262865556059567i,...
                -0.138196601125011 + 0.425325404176020i;
                -0.361803398874989 - 0.262865556059567i,...
                 0.138196601125011 - 0.425325404176020i,...
                 0.447213595499958 + 0.000000000000000i,...
                 0.138196601125010 + 0.425325404176020i,...
                -0.361803398874989 + 0.262865556059567i
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            coefExpctd = [...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            ang = 2*pi*rand(10+6+2+2,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        function testConstructorWithDec4Ch5Ord4Ang(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 4;
            ang = 2*pi*rand(10+16+4,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        
        % Test for construction with order 2
        function testConstructorWithDec5Ord2(testCase)
            
            % Parameters
            dec = 5;
            ord = 2;
            ang = 2*pi*rand(10+8+2,1);
            
            % Expected values
            nDec = dec;
            nChs = nDec;
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4
        function testConstructorWithDec5Ord4(testCase)
            
            % Parameters
            dec = 5;
            ord = 4;
            ang = 2*pi*rand(10+16+4,1);
            
            % Expected values
            nDec = dec;
            nChs = nDec;
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord+1) = ...
                coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec4Ch7Ord0(testCase)
            
            % Parameters
            decch = [ 4 7 ];
            ord   = 0;
            
            % Expected values
            dimExpctd = [7 4];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            
            % Check tightness
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            
        end
        
        % Test for construction
        function testConstructorWithDec4Ch9Ord0(testCase)
            
            % Parameters
            decch = [ 4 9 ];
            ord   = 0;
            
            % Expected values
            dimExpctd = [9 4];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec5Ch9Ord0Ang0(testCase)
            
            % Parameters
            decch = [ 5 9 ];
            ord   = 0;
            ang   = 0;
            
            % Expected values
            coefExpctd = [...
                 0.447213595499958 + 0.000000000000000i,...
                 0.447213595499958 + 0.000000000000000i,...
                 0.447213595499958 + 0.000000000000000i,...
                 0.447213595499958 + 0.000000000000000i,...
                 0.447213595499958 + 0.000000000000000i;
                 0.361803398874989 - 0.262865556059567i,...
                -0.138196601125011 - 0.425325404176020i,...
                -0.447213595499958 - 0.000000000000000i,...
                -0.138196601125011 + 0.425325404176020i,...
                 0.361803398874989 + 0.262865556059567i;
                 0.138196601125011 - 0.425325404176020i,...
                -0.361803398874989 + 0.262865556059567i,...
                 0.447213595499958 + 0.000000000000000i,...
                -0.361803398874989 - 0.262865556059567i,...
                 0.138196601125010 + 0.425325404176020i;
                -0.138196601125011 - 0.425325404176020i,...
                 0.361803398874989 + 0.262865556059567i,...
                -0.447213595499958 - 0.000000000000000i,...
                 0.361803398874989 - 0.262865556059567i,...
                -0.138196601125011 + 0.425325404176020i;
                -0.361803398874989 - 0.262865556059567i,...
                 0.138196601125011 - 0.425325404176020i,...
                 0.447213595499958 + 0.000000000000000i,...
                 0.138196601125010 + 0.425325404176020i,...
                -0.361803398874989 + 0.262865556059567i;
                0 0 0 0 0 
                0 0 0 0 0 
                0 0 0 0 0                 
                0 0 0 0 0                                 
                ];
                
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,...
                sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec9Ch9Ord0Ang(testCase)
            
            % Parameters
            decch = [ 9 9 ];
            ord = 0;
%             angW = zeros(10,1);
%             angU = 2*pi*rand(6,1);
            angV0 = 2*pi*rand(36,1);
            
            % Expected values
            import saivdr.dictionary.utility.*
%             omgsW = OrthonormalMatrixGenerationSystem();
%             omgsU = OrthonormalMatrixGenerationSystem();
%             matrixW0 = step(omgsW,angW,1);
%             matrixU0 = step(omgsU,angU,1);
            omgsV0 = OrthonormalMatrixGenerationSystem();
            matrixV0 = step(omgsV0,angV0,1);
            coefExpctd = ...
                matrixV0 * ...
                [ 0.333333333333333  + 0.000000000000000i, 0.333333333333333 + 0.000000000000000i, 0.333333333333333  + 0.000000000000000i, 0.333333333333333  + 0.000000000000000i, 0.333333333333333 + 0.000000000000000i, 0.333333333333333  + 0.000000000000000i,  0.333333333333333  + 0.000000000000000i, 0.333333333333333 + 0.000000000000000i, 0.333333333333333  + 0.000000000000000i;
                  0.313230873595303  - 0.114006714441890i, 0.166666666666667 - 0.288675134594813i,-0.0578827258889767 - 0.328269251004069i,-0.255348147706326  - 0.214262536562180i,-0.333333333333333 + 0.000000000000000i,-0.255348147706326  + 0.214262536562180i, -0.0578827258889768 + 0.328269251004069i, 0.166666666666667 + 0.288675134594813i, 0.313230873595303  + 0.114006714441890i;
                  0.255348147706326  - 0.214262536562180i,-0.166666666666667 - 0.288675134594813i,-0.313230873595303  + 0.114006714441889i, 0.0578827258889767 + 0.328269251004069i, 0.333333333333333 + 0.000000000000000i, 0.0578827258889769 - 0.328269251004069i, -0.313230873595303  - 0.114006714441890i,-0.166666666666667 + 0.288675134594813i, 0.255348147706326  + 0.214262536562180i;
                  0.166666666666667  - 0.288675134594813i,-0.333333333333333 - 0.000000000000000i, 0.166666666666667  + 0.288675134594813i, 0.166666666666667  - 0.288675134594813i,-0.333333333333333 + 0.000000000000000i, 0.166666666666666  + 0.288675134594813i,  0.166666666666667  - 0.288675134594812i,-0.333333333333333 + 0.000000000000000i, 0.166666666666666  + 0.288675134594813i;
                  0.0578827258889768 - 0.328269251004069i,-0.166666666666667 + 0.288675134594813i, 0.255348147706326  - 0.214262536562180i,-0.313230873595303  + 0.114006714441889i, 0.333333333333333 + 0.000000000000000i,-0.313230873595302  - 0.114006714441890i,  0.255348147706325  + 0.214262536562180i,-0.166666666666666 - 0.288675134594813i, 0.0578827258889762 + 0.328269251004069i;
                 -0.0578827258889767 - 0.328269251004069i, 0.166666666666667 + 0.288675134594813i,-0.255348147706326  - 0.214262536562180i, 0.313230873595302  + 0.114006714441890i,-0.333333333333333 + 0.000000000000000i, 0.313230873595302  - 0.114006714441889i, -0.255348147706326  + 0.214262536562179i, 0.166666666666667 - 0.288675134594812i,-0.0578827258889772 + 0.328269251004068i;
                 -0.166666666666667  - 0.288675134594813i, 0.333333333333333 + 0.000000000000000i,-0.166666666666667  + 0.288675134594813i,-0.166666666666666  - 0.288675134594813i, 0.333333333333333 + 0.000000000000000i,-0.166666666666667  + 0.288675134594813i, -0.166666666666666  - 0.288675134594812i, 0.333333333333332 + 0.000000000000000i,-0.166666666666667  + 0.288675134594812i;
                 -0.255348147706326  - 0.214262536562180i, 0.166666666666667 - 0.288675134594813i, 0.313230873595302  + 0.114006714441889i,-0.0578827258889770 + 0.328269251004069i,-0.333333333333333 + 0.000000000000000i,-0.0578827258889761 - 0.328269251004069i,  0.313230873595302  - 0.114006714441889i, 0.166666666666667 + 0.288675134594812i,-0.255348147706326  + 0.214262536562178i;
                 -0.313230873595303  - 0.114006714441890i,-0.166666666666666 - 0.288675134594813i, 0.0578827258889770 - 0.328269251004069i, 0.255348147706326  - 0.214262536562179i, 0.333333333333333 + 0.000000000000000i, 0.255348147706325  + 0.214262536562180i,  0.0578827258889759 + 0.328269251004068i,-0.166666666666667 + 0.288675134594811i,-0.313230873595302  + 0.114006714441888i...
                 ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            %coefActual = step(testCase.lppufb,[angW;angU],[]);
            coefActual = step(testCase.lppufb,angV0,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,...
                sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec4Ch5Ord0Ang(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 0;
            ang = 2*pi*rand(10,1);
            
            % Expected values
            dimExpctd = [5 4];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            
        end

        % Test for construction
        function testConstructorWithDec4Ch7Ord0Ang(testCase)
            
            % Parameters
            decch = [ 4 7 ];
            ord   = 0;
            ang = 2*pi*rand(21,1);
            
            % Expected values
            dimExpctd = [7 4];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec4Ch9Ord0Ang(testCase)
            
            % Parameters
            decch = [ 4 9 ];
            ord   = 0;
            ang = 2*pi*rand(36,1);
            
            % Expected values
            dimExpctd = [9 4];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec9Ch11Ord0Ang(testCase)
            
            % Parameters
            decch = [ 9 11 ];
            ord   = 0;
            ang = 2*pi*rand(55,1);
            
            % Expected values
            dimExpctd = [11 9];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec1Ch5Ord0(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord   = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [...
                1;
                0;
                0;
                0;
                0];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec1Ch5Ord0Ang0(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord   = 0;
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                1 ;
                0 ;
                0 ;
                0 ;
                0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec1Ch5Ord0AngPi3(testCase)
            %TODO: テストの内容を確認する
            % Parameters
            decch = [ 1 5 ];
            ord   = 0;
%             angW = zeros(3,1);
%             angU = pi/3;
            angV0 = pi/3*ones(10,1);
            
            % Expected values
            import saivdr.dictionary.utility.*
%             omgsW = OrthonormalMatrixGenerationSystem();
%             omgsU = OrthonormalMatrixGenerationSystem();
%             matrixW0 = step(omgsW,angW,1);
%             matrixU0 = step(omgsU,angU,1);
            omgsV0 = OrthonormalMatrixGenerationSystem();
            matrixV0 = step(omgsV0,angV0,1);
            
            coefExpctd(:,:,1) = ...
                matrixV0 * ...
                [ 1 0 0 0 0 ].';
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,angV0,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Invalid arguments
        function testConstructorWithInvalidArguments(testCase)
            
            % Invalid input
            decch = [ 4 5 ];
            ord   = 0;
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
                import saivdr.dictionary.colpprfb.*
                testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                    'DecimationFactor',decch(1),...
                    'NumberOfChannels',decch(2:end),...
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
            decch = [ 4 5 ];
            ord   = 0;
            ang = zeros(10,1);
            mus = [ 1 1 1 1 1 ].';
            
            % Expected values
            coefExpctd(:,:,1) = diag(mus)*[...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 2
        % テスト名を修正する
        function testConstructorWithDec4Ch5Ord4Ang0(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord   = 4;
            ang   = [0 0 0 0 0 0 0 0 0 0 0 0 pi/4 0 0 0 0 0 0 pi/4 0 0 pi/4 0 0 0 0 0 0 pi/4];
            
            % Expected values
            coefExpctd = zeros(5,4,5);
            coefExpctd(:,:,3) = [...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 8
        function testConstructorWithDec4Ch5Ord8Ang(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord   = 8;
            ang = 2*pi*rand(10+32+8,1);
            
            % Expected values
            nDecs = decch(1);
            dimExpctd = [decch(2) nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test dec 9 order 4
        function testConstructorWithDec9Ch9Ord4Ang0(testCase)
            
            % Parameters
            decch = [ 9 9 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(9,9,5);
            coefExpctd(:,:,3) = ...
                [ 0.333333333333333  + 0.000000000000000i, 0.333333333333333 + 0.000000000000000i, 0.333333333333333  + 0.000000000000000i, 0.333333333333333  + 0.000000000000000i, 0.333333333333333 + 0.000000000000000i, 0.333333333333333  + 0.000000000000000i,  0.333333333333333  + 0.000000000000000i, 0.333333333333333 + 0.000000000000000i, 0.333333333333333  + 0.000000000000000i;
                  0.313230873595303  - 0.114006714441890i, 0.166666666666667 - 0.288675134594813i,-0.0578827258889767 - 0.328269251004069i,-0.255348147706326  - 0.214262536562180i,-0.333333333333333 + 0.000000000000000i,-0.255348147706326  + 0.214262536562180i, -0.0578827258889768 + 0.328269251004069i, 0.166666666666667 + 0.288675134594813i, 0.313230873595303  + 0.114006714441890i;
                  0.255348147706326  - 0.214262536562180i,-0.166666666666667 - 0.288675134594813i,-0.313230873595303  + 0.114006714441889i, 0.0578827258889767 + 0.328269251004069i, 0.333333333333333 + 0.000000000000000i, 0.0578827258889769 - 0.328269251004069i, -0.313230873595303  - 0.114006714441890i,-0.166666666666667 + 0.288675134594813i, 0.255348147706326  + 0.214262536562180i;
                  0.166666666666667  - 0.288675134594813i,-0.333333333333333 - 0.000000000000000i, 0.166666666666667  + 0.288675134594813i, 0.166666666666667  - 0.288675134594813i,-0.333333333333333 + 0.000000000000000i, 0.166666666666666  + 0.288675134594813i,  0.166666666666667  - 0.288675134594812i,-0.333333333333333 + 0.000000000000000i, 0.166666666666666  + 0.288675134594813i;
                  0.0578827258889768 - 0.328269251004069i,-0.166666666666667 + 0.288675134594813i, 0.255348147706326  - 0.214262536562180i,-0.313230873595303  + 0.114006714441889i, 0.333333333333333 + 0.000000000000000i,-0.313230873595302  - 0.114006714441890i,  0.255348147706325  + 0.214262536562180i,-0.166666666666666 - 0.288675134594813i, 0.0578827258889762 + 0.328269251004069i;
                 -0.0578827258889767 - 0.328269251004069i, 0.166666666666667 + 0.288675134594813i,-0.255348147706326  - 0.214262536562180i, 0.313230873595302  + 0.114006714441890i,-0.333333333333333 + 0.000000000000000i, 0.313230873595302  - 0.114006714441889i, -0.255348147706326  + 0.214262536562179i, 0.166666666666667 - 0.288675134594812i,-0.0578827258889772 + 0.328269251004068i;
                 -0.166666666666667  - 0.288675134594813i, 0.333333333333333 + 0.000000000000000i,-0.166666666666667  + 0.288675134594813i,-0.166666666666666  - 0.288675134594813i, 0.333333333333333 + 0.000000000000000i,-0.166666666666667  + 0.288675134594813i, -0.166666666666666  - 0.288675134594812i, 0.333333333333332 + 0.000000000000000i,-0.166666666666667  + 0.288675134594812i;
                 -0.255348147706326  - 0.214262536562180i, 0.166666666666667 - 0.288675134594813i, 0.313230873595302  + 0.114006714441889i,-0.0578827258889770 + 0.328269251004069i,-0.333333333333333 + 0.000000000000000i,-0.0578827258889761 - 0.328269251004069i,  0.313230873595302  - 0.114006714441889i, 0.166666666666667 + 0.288675134594812i,-0.255348147706326  + 0.214262536562178i;
                 -0.313230873595303  - 0.114006714441890i,-0.166666666666666 - 0.288675134594813i, 0.0578827258889770 - 0.328269251004069i, 0.255348147706326  - 0.214262536562179i, 0.333333333333333 + 0.000000000000000i, 0.255348147706325  + 0.214262536562180i,  0.0578827258889759 + 0.328269251004068i,-0.166666666666667 + 0.288675134594811i,-0.313230873595302  + 0.114006714441888i...
                 ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,...
                sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 order 4
        function testConstructorWithDec4Ch7Ord4Ang(testCase)
            
            % Parameters
            decch = [ 4 7 ];
            ord = 4;
            ang = 2*pi*rand(21+36+4,1);
            
            % Expected values
            nDecs = decch(1);
            nChs = decch(2);
            dimExpctd = [nChs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test: dec 9 order 8
        function testConstructorWithDecCh9Ord8Ang(testCase)
            
            % Parameters
            decch = [ 9 9 ];
            ord = 8;
            ang = 2*pi*rand(36+4*32+16,1);
            
            % Expected values
            nDecs = decch(1);
            nChs = decch(2);
            dimExpctd = [nChs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),...
                sprintf('%g',coefDist));
            
        end
        
        % Test for angle setting
        function testSetAngles(testCase)
            %TODO: テストケースの内容修正
            % Parameters
            decch = [ 4 5 ];
            ord = 0;
            angPre = [ pi/4 pi/4 pi/4 pi/4 pi/4 pi/4 pi/4 pi/4 pi/4 pi/4].';
            angPst = [ 0 0 0 0 0 0 0 0 0 0 ].';
            
            % Expected values
            coefExpctd = [...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,angPre,[]);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThanOrEqualTo;
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThanOrEqualTo(1e-14),...
                sprintf('%g',coefDist));
            
            % Actual values
            coefActual = step(testCase.lppufb,angPst,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,...
                sprintf('%g',coefDist));
            
        end
        
        % Test for mus setting
        function testSetMus(testCase)
            %TODO: テストの必要性を検討
            % Parameters
            decch = [ 4 5 ];
            ord = 0;
            ang = [ 0 0 0 0 0 0 0 0 0 0 ].';
            musPre = [ 1 -1  1 -1 1 -1 1 -1 1 -1].';
            musPst = 1;
            
            % Expected values
            coefExpctd = [...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
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
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,...
                sprintf('%g',coefDist));
            
        end
        
        % Test for subsref
        function testAnalysisFilterAt(testCase)
            
            % Expected value
            anFiltExpctd1 = [ 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i ];
            anFiltExpctd2 = [ 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i ];
            anFiltExpctd3 = [ 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i ];
            anFiltExpctd4 = [-0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i ];
            anFiltExpctd5 = [ 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            anFiltExpctd1 = [ 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i ];
            anFiltExpctd2 = [ 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i ];
            anFiltExpctd3 = [ 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i ];
            anFiltExpctd4 = [-0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i ];
            anFiltExpctd5 = [ 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'OutputMode','AnalysisFilters');
            
            % Actual values
            anFiltsActual = step(testCase.lppufb,[],[]);
            anFiltActual1 = anFiltsActual(:,1);
            anFiltActual2 = anFiltsActual(:,2);
            anFiltActual3 = anFiltsActual(:,3);
            anFiltActual4 = anFiltsActual(:,4);
            anFiltActual5 = anFiltsActual(:,5);            
            
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

        % Test dec 4 ch 5 order 2
        function testConstructorWithDec4Ch5Ord2(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 2;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(5,4,3);
            coefExpctd(:,:,2) = [...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 ch 5 order 2
        function testConstructorWithDec1Ch5Ord2(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord = 2;
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0  ;
                0  ;
                0  ;
                0  ;
                0  ];
            
            coefExpctd(:,:,2) = [
                1 ;
                0 ;
                0 ;
                0 ;
                0 ];
            
            coefExpctd(:,:,3) = [
                0  ;
                0  ;
                0  ;
                0  ;
                0  ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 ch 5 order 4
        function testConstructorWithDec1Ch5Ord4(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2) =  [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3) = [...
                1;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,4) =  [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,5) = [...
                0;
                0;
                0;
                0;
                0];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        function testConstructorWithDec4Ch5Ord4(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(5,4,5);
            coefExpctd(:,:,3) = [...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 order 8
        function testConstructorWithDec4Ch5Ord8(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 8;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(5,4,9);
            coefExpctd(:,:,5) = [...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
         % Test dec 4 ch 5 order 12
        function testConstructorWithDec4Ch5Ord12Ang(testCase)
            
            % Parameters
            decch = [ 4 5 ];
            ord = 12;
            ang = 2*pi*rand(10+48+12,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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

        % Test dec 4 ch 9 order 4
        function testConstructorWithDec4Ch9Ord4Ang(testCase)
            
            % Parameters
            decch = [ 4 9 ];
            ord = 4;
            ang = 2*pi*rand(36+64+8,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        
        % Test dec 4 ch 9 order 8
        function testConstructorWithDec4Ch9Ord8Ang(testCase)
            
            % Parameters
            decch = [ 4 9 ];
            ord = 8;
            ang = 2*pi*rand(36+4*32+16,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        
        % Test dec 4 ch 11 order 4
        function testConstructorWithDec4Ch11Ord4Ang(testCase)
            
            % Parameters
            decch = [ 4 11 ];
            ord = 4;
            ang = 2*pi*rand(55+100+8,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        
        % Test dec 6 ch 11 order 4
        function testConstructorWithDec6Ch11Ord4Ang(testCase)
            
            % Parameters
            decch = [ 6 11 ];
            ord = 4;
            ang = 2*pi*rand(55+100+8,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        
        % Test dec 6 ch 11 order 8
        function testConstructorWithDec6Ch11Ord8Ang(testCase)
            
            % Parameters
            decch = [ 6 11 ];
            ord = 8;
            ang = 2*pi*rand(55+200+16,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        
        % Test dec 1 ch 5 order 4
        function testConstructorWithDec1Ch5Ord4Ang(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord = 4;
            ang = 2*pi*rand(10+16+4,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        
        % Test dec 1 ch 5 order 8
        function testConstructorWithDec1Ch5Ord8Ang(testCase)
            
            % Parameters
            decch = [ 1 5 ];
            ord = 8;
            ang = 2*pi*rand(10+32+8,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        
        % Test dec 1 ch 7 order 8
        function testConstructorWithDec1Ch7Ord8Ang(testCase)
            
            % Parameters
            decch = [ 1 7 ];
            ord = 8;
            ang = 2*pi*rand(21+4*18+8,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        
        % Test dec 1 ch 7 order 12
        function testConstructorWithDec1Ch7Ord12Ang(testCase)
            
            % Parameters
            decch = [ 1 7 ];
            ord = 12;
            ang = 2*pi*rand(21+6*18+12,1);
            
            % Expected values
            nChs = decch(2);
            nDec = decch(1);
            dimExpctd = [nChs nDec ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        
        % Test for construction
        function testConstructorWithDec4Ch32Ord0(testCase)
            
            % Parameters
            decch = [ 4 3 2 ];
            ord = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
%         % Test for construction
%         function testConstructorWithDec4Ch42Ord0(testCase)
%             
%             % Parameters
%             decch = [ 4 4 2 ];
%             ord = 0;
%             
%             % Expected values
%             coefExpctd(:,:,1) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0 0 0 0
%                 0 0 0 0
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
%                 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
        
        % Test for construction
        function testConstructorWithDec4Ch43Ord0(testCase)
            
            % Parameters
            decch = [ 4 4 3 ];
            ord = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
%         % Test for construction
%         function testConstructorWithDec4Ch52Ord0(testCase)
%             
%             % Parameters
%             decch = [ 4 5 2 ];
%             ord = 0;
%             
%             % Expected values
%             coefExpctd(:,:,1) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0 0 0 0
%                 0 0 0 0
%                 0 0 0 0                
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
%                 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test for construction
%         function testConstructorWithDec4Ch62Ord0(testCase)
%             
%             % Parameters
%             decch = [ 4 6 2 ];
%             ord = 0;
%             
%             % Expected values
%             coefExpctd(:,:,1) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0 0 0 0
%                 0 0 0 0
%                 0 0 0 0                
%                 0 0 0 0                        
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
%                 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end

        % Test for construction with order 4
        function testConstructorWithDec4Ch32Ord4(testCase)
            
            % Parameters
            decch = [ 4 3 2 ];
            ord = 4;
            ang = 0;
            
            % Expected values
            coefExpctd  = zeros(5,4,5);
            
            coefExpctd(:,:,3) = [...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
%         % Test for construction with order 4
%         function testConstructorWithDec4Ch42Ord4(testCase)
%             
%             % Parameters
%             decch = [ 4 4 2 ];
%             ord = 4;
%             ang = 0;
%             
%             % Expected values
%             coefExpctd = zeros(6,4,5);
%             coefExpctd(:,:,3) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0 0 0 0
%                 0 0 0 0
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
%                 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 4
%         function testConstructorWithDec4Ch52Ord4(testCase)
%             
%             % Parameters
%             decch = [ 4 5 2 ];
%             ord = 4;
%             ang = 0;
%             
%             % Expected values
%             coefExpctd = zeros(7,4,5);
%             coefExpctd(:,:,3) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0 0 0 0
%                 0 0 0 0
%                 0 0 0 0
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
%                 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 4
%         function testConstructorWithDec4Ch53Ord4(testCase)
%             
%             % Parameters
%             decch = [ 4 5 3 ];
%             ord = 4;
%             ang = 0;
%             
%             % Expected values
%             coefExpctd = zeros(8,4,5);
%             coefExpctd(:,:,3) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0 0 0 0
%                 0 0 0 0
%                 0 0 0 0
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
%                 0 0 0 0
%                 ];            
% 
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test for construction with order 4
%         function testConstructorWithDec4Ch62Ord4(testCase)
%             
%             % Parameters
%             decch = [ 4 6 2 ];
%             ord = 4;
%             ang = 0;
%             
%             % Expected values
%             coefExpctd = zeros(8,4,5);
%             coefExpctd(:,:,3) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0 0 0 0
%                 0 0 0 0
%                 0 0 0 0
%                 0 0 0 0
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
%                 ];            
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end

        % Test dec 4 order 8
        function testConstructorWithDec4Ch32Ord8(testCase)
            
            % Parameters
            decch = [ 4 3 2 ];
            ord = 8;
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(5,4,9);
            coefExpctd(:,:,5) = [...
                 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i, 0.500000000000000 + 0.000000000000000i;
                 0.353553390593274 - 0.353553390593274i,-0.353553390593274 - 0.353553390593274i,-0.353553390593274 + 0.353553390593274i, 0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i, 0.000000000000000 - 0.500000000000000i, 0.000000000000000 + 0.500000000000000i;
                -0.353553390593274 - 0.353553390593274i, 0.353553390593274 - 0.353553390593274i, 0.353553390593274 + 0.353553390593274i,-0.353553390593274 + 0.353553390593274i;
                 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i, 0.000000000000000 + 0.000000000000000i...
                ];            
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'DecimationFactor',decch(1),...
                'NumberOfChannels',decch(2:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end     

%         % Test dec 4 order 8
%         function testConstructorWithDec4Ch42Ord8(testCase)
%             
%             % Parameters
%             decch = [ 4 4 2 ];
%             ord = 8;
%             ang = 0;
%             
%             % Expected values
%             coefExpctd = zeros(6,4,9);
%             coefExpctd(:,:,5) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0 0 0 0
%                 0 0 0 0
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099 
%                 ];                        
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end  
        
        % Test dec 4 order 4
        function testConstructorWithDec4Ch32Ord4Ang(testCase)
            
          % Parameters
            decch = [ 4 3 2 ];
            ord = 4;
            ang = 2*pi*rand(10+16+4,1);
            
            % Expected values
            nDec = prod(decch(1));
            nChs = sum(decch(2:end));
            dimExpctd = [nChs nDec ord+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
        
        % Test dec 4 order 8
        function testConstructorWithDec4Ch32Ord8Ang(testCase)
            
            % Parameters
            decch = [ 4 3 2 ];
            ord = 8;
            ang = 2*pi*rand(10+32+8,1);
            
            % Expected values
            nDecs = decch(1);
            nChs = sum(decch(2:end));
            dimExpctd = [nChs nDecs ord+1];
            
            % Instantiation of target class
            import saivdr.dictionary.colpprfb.*
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
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
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDecs,1:nDecs,ord+1) = ...
                coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
%         % Test dec 4 order 4 
%         function testConstructorWithDec4Ch42Ord4Ang(testCase)
%             
%           % Parameters
%             decch = [ 4 4 2 ];
%             ord = 4;
%             ang = 2*pi*rand(7,3);
%             
%             % Expected values
%             nDec = prod(decch(1));
%             nChs = sum(decch(2:end));
%             dimExpctd = [nChs nDec ord+1 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));  
%             
%             % Check tightness
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDec,1:nDec,ord+1) = ...
%                 coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
%             coefDist = max(abs(coefActual(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
%         
%         % Test dec 4 order 8
%         function testConstructorWithDec4Ch42Ord8Ang(testCase)
%             
%             % Parameters
%             decch = [ 4 4 2 ];
%             ord = 8;
%             ang = 2*pi*rand(7,5);
%             
%             % Expected values
%             nDecs = decch(1);
%             nChs = sum(decch(2:end));
%             dimExpctd = [nChs nDecs ord+1];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));  
%             
%             % Check orthogonality
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDecs,1:nDecs,ord+1) = ...
%                 coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
%             coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
       
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
            testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end
        
%         % Test for construction
%         function testConstructorWithDec4Ch24Ord0(testCase)
%             
%             % Parameters
%             decch = [ 4 2 4 ];
%             ord = 0;
%             
%             % Expected values
%             coefExpctd = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
%                 0 0 0 0
%                 0 0 0 0
%                 ];
%         
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test for construction
%         function testConstructorWithDec4Ch34Ord0(testCase)
%             
%             % Parameters
%             decch = [ 4 3 4 ];
%             ord = 0;
%             
%             % Expected values
%             coefExpctd = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0 0 0 0                
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
%                 0 0 0 0
%                 0 0 0 0
%                 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test for construction
%         function testConstructorWithDec4Ch25Ord0(testCase)
%             
%             % Parameters
%             decch = [ 4 2 5 ];
%             ord = 0;
%             
%             % Expected values
%             coefExpctd = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
%                 0 0 0 0
%                 0 0 0 0 
%                 0 0 0 0 
%                 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test for construction
%         function testConstructorWithDec4Ch26Ord0(testCase)
%             
%             % Parameters
%             decch = [ 4 2 6 ];
%             ord = 0;
%             
%             % Expected values
%             coefExpctd = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
%                 0 0 0 0 
%                 0 0 0 0 
%                 0 0 0 0 
%                 0 0 0 0
%                 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 4
%         function testConstructorWithDec4Ch23Ord4(testCase)
%             
%             % Parameters
%             decch = [ 4 2 3 ];
%             ord = 4;
%             ang = 0;
%             
%             % Expected values
%             coefExpctd = zeros(5,4,5);
%             coefExpctd(:,:,3) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
%                 0 0 0 0 
%                 ];
% 
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test for construction with order 4
%         function testConstructorWithDec4Ch24Ord4(testCase)
%             
%             % Parameters
%             decch = [ 4 2 4 ];
%             ord = 4;
%             ang = 0;
%             
%             % Expected values
%             coefExpctd = zeros(6,4,5);
%             coefExpctd(:,:,3) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
%                 0 0 0 0 
%                 0 0 0 0                 
%                 ];
% 
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test for construction with order 4
%         function testConstructorWithDec4Ch25Ord4(testCase)
%             
%             % Parameters
%             decch = [ 4 2 5 ];
%             ord = 4;
%             ang = 0;
%             
%             % Expected values
%             coefExpctd = zeros(7,4,5);
%             coefExpctd(:,:,3) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
%                 0 0 0 0 
%                 0 0 0 0 
%                 0 0 0 0                 
%                 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 4
%         function testConstructorWithDec4Ch35Ord4(testCase)
%             
%             % Parameters
%             decch = [ 4 3 5 ];
%             ord = 4;
%             ang = 0;
%             
%             % Expected values
%             coefExpctd = zeros(8,4,5);
%             coefExpctd(:,:,3) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0 0 0 0 
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
%                 0 0 0 0 
%                 0 0 0 0 
%                 0 0 0 0                 
%                 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test for construction with order 4
%         function testConstructorWithDec4Ch26Ord4(testCase)
%             
%             % Parameters
%             decch = [ 4 2 6 ];
%             ord = 4;
%             ang = 0;
%             
%             % Expected values
%             coefExpctd = zeros(8,4,5);
%             coefExpctd(:,:,3) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
%                 0 0 0 0 
%                 0 0 0 0 
%                 0 0 0 0 
%                 0 0 0 0                 
%                 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test dec 4 order 8
%         function testConstructorWithDec4Ch23Ord8(testCase)
%             
%             % Parameters
%             decch = [ 4 2 3 ];
%             ord = 8;  
%             ang = 0;
%             
%             % Expected values
%             coefExpctd = zeros(5,4,9);
%             coefExpctd(:,:,5) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
%                 0 0 0 0                 
%                 ];
% 
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end     
% 
%         % Test dec 4 order 8
%         function testConstructorWithDec4Ch24Ord8(testCase)
%             
%             % Parameters
%             decch = [ 4 2 4 ];
%             ord = 8;
%             ang = 0;
%             
%             % Expected values
%             coefExpctd = zeros(6,4,9);
%             coefExpctd(:,:,5) = [
%                 0.500000000000000   0.500000000000000   0.500000000000000   0.500000000000000
%                 0.500000000000000  -0.500000000000000  -0.500000000000000   0.500000000000000
%                 0.653281482438188   0.270598050073099  -0.270598050073099  -0.653281482438188
%                 0.270598050073099  -0.653281482438188   0.653281482438188  -0.270598050073099
%                 0 0 0 0                 
%                 0 0 0 0                 
%                 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end  
%         
%         % Test dec 4 order 4
%         function testConstructorWithDec4Ch23Ord4Ang(testCase)
%             
%           % Parameters
%             decch = [ 4 2 3 ];
%             ord = 4;
%             ang = 2*pi*rand(4,3);
%             
%             % Expected values
%             nDec = prod(decch(1));
%             nChs = sum(decch(2:end));
%             dimExpctd = [nChs nDec ord+1 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));   
%             
%             % Check tightness
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDec,1:nDec,ord+1) = ...
%                 coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
%             coefDist = max(abs(coefActual(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
% 
%         % Test dec 4 order 8
%         function testConstructorWithDec4Ch23Ord8Ang(testCase)
%             
%             % Parameters
%             decch = [ 4 2 3 ];
%             ord = 8;
%             ang = 2*pi*rand(4,5);
%             
%             % Expected values
%             nDecs = decch(1);
%             nChs = sum(decch(2:end));
%             dimExpctd = [nChs nDecs ord+1];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
%             
%             % Check orthogonality
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDecs,1:nDecs,ord+1) = ...
%                 coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
%             coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
% 
%         % Test dec 4 order 4
%         function testConstructorWithDec4Ch24Ord4Ang(testCase)
%             
%           % Parameters
%             decch = [ 4 2 4 ];
%             ord = 4;
%             ang = 2*pi*rand(7,3);
%             
%             % Expected values
%             nDec = prod(decch(1));
%             nChs = sum(decch(2:end));
%             dimExpctd = [nChs nDec ord+1 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));  
%             
%             % Check tightness
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDec,1:nDec,ord+1) = ...
%                 coefActual(1:nDec,1:nDec,ord+1) - eye(nDec);
%             coefDist = max(abs(coefActual(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
%         
%         % Test dec 4 order 8
%         function testConstructorWithDec4Ch24Ord8Ang(testCase)
%             
%             % Parameters
%             decch = [ 4 2 4 ];
%             ord = 8;
%             ang = 2*pi*rand(7,5);
%             
%             % Expected values
%             nDecs = decch(1);
%             nChs = sum(decch(2:end));
%             dimExpctd = [nChs nDecs ord+1];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'DecimationFactor',decch(1),...
%                 'NumberOfChannels',decch(2:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));  
%             
%             % Check orthogonality
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix1d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDecs,1:nDecs,ord+1) = ...
%                 coefActual(1:nDecs,1:nDecs,ord+1) - eye(nDecs);
%             coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
%        
%         function testParameterMatrixSetCh23(testCase)
%             
%             % Preparation
%             nchs  = [ 2 3 ];
%             mstab = [ 2 2 ; 3 3 ];
%             
%             % Expected value
%             import saivdr.dictionary.utility.ParameterMatrixSet
%             paramExpctd = ParameterMatrixSet(...
%                 'MatrixSizeTable',mstab);
%             step(paramExpctd,eye(2),1);
%             step(paramExpctd,eye(3),2);
%             
%             % Instantiation of target class
%             import saivdr.dictionary.colpprfb.*
%             testCase.lppufb = CplxOvsdLpPuFb1dTypeIIVm0System(...
%                 'OutputMode','ParameterMatrixSet',...
%                 'NumberOfChannels',nchs);
%             
%             % Actual values
%             paramActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             testCase.verifyEqual(paramExpctd, paramActual);
%             
%         end

    end
    
end
