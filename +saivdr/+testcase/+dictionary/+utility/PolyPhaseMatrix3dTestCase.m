classdef PolyPhaseMatrix3dTestCase < matlab.unittest.TestCase
    %POLYPHASEMATRIX3DTESTCASE Test case for PolyPhaseMatrix3d
    %
    % SVN identifier:
    % $Id: PolyPhaseMatrix3dTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        ppm0
        ppm1
        ppm2
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.ppm0)
            delete(testCase.ppm1)
            delete(testCase.ppm2)
        end
    end
    
    methods (Test)
        
        % Test for default construction
        function testConstructor(testCase)
            
            % Expected values
            coefsExpctd = [];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d();
            
            % Actual values
            coefsActual = double(testCase.ppm0);
            
            % Evaluation
            testCase.verifyEqual(coefsActual,coefsExpctd);
            
        end

        % Test for construction with initialization
        function testConstructorWithInit(testCase)
            
            % Input coeffficients
            coefs(:,:,1) = [
                1 3
                2 4 ];
            coefs(:,:,2) = [
                5 6
                7 8 ];            
            
            % Expected values
            coefsExpctd = coefs;
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coefs);
            
            % Actual values
            coefsActual = double(testCase.ppm0);
            
            % Evaluation
            testCase.verifyEqual(coefsActual,coefsExpctd);
            
        end
        
        % Test for object construction
        function testConstructorWithObj(testCase)
            
            % Input value
            coefs(:,:,1) = [
                1 3 ;
                2 4 ];
            coefs(:,:,2) = [
                5 6;
                7 8 ];
            
            % Expected value
            coefsExpctd = coefs;
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coefs);
            testCase.ppm1 = PolyPhaseMatrix3d(testCase.ppm0);
            
            % Actual values
            coefsActual = double(testCase.ppm1);
            
            % Evaluation
            testCase.verifyEqual(coefsActual,coefsExpctd);
            
        end   
        
        % Test for subsref
        function testSubsRef(testCase)
            
            % Input value
            % 0 0 0
            coefs(:,:,1,1,1) = [
                1 1 ;
                1 1 ];
            % 1 0 0            
            coefs(:,:,2,1,1) = [
                2 2 ;
                -2 2 ];
            % 0 1 0                        
            coefs(:,:,1,2,1) = [
                3 -3 ;
                3 3 ];
            % 1 1 0
            coefs(:,:,2,2,1) = [
                4 4 ;
                4 -4 ];
            % 0 0 1
            coefs(:,:,1,1,2) = [
                5 5 ;
                5 5 ];
            % 1 0 1                                                
            coefs(:,:,2,1,2) = [
                6 6 ;
                6 6 ];
            % 0 1 1                                                
            coefs(:,:,1,2,2) = [
                7 7 ;
                7 7 ];
            % 1 1 1                                                
            coefs(:,:,2,2,2) = [
                8 8 ;
                8 8 ];
            
            % Expected value
            ppfExpctd11(:,:,1) = [ 1 3; 2 4 ];
            ppfExpctd11(:,:,2) = [ 5 7; 6 8 ];            
            ppfExpctd21(:,:,1) = [ 1 3; -2 4 ];
            ppfExpctd21(:,:,2) = [ 5 7; 6 8 ];                                    
            ppfExpctd12(:,:,1) = [ 1 -3; 2 4 ];
            ppfExpctd12(:,:,2) = [ 5 7; 6 8 ];                                                
            ppfExpctd22(:,:,1) = [ 1 3; 2 -4 ];
            ppfExpctd22(:,:,2) = [ 5 7; 6 8 ];                                                
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*
            testCase.ppm0 = PolyPhaseMatrix3d(coefs);
            
            % Actual values
            ppfActual11 = testCase.ppm0(1,1);
            ppfActual21 = testCase.ppm0(2,1);            
            ppfActual12 = testCase.ppm0(1,2);
            ppfActual22 = testCase.ppm0(2,2);
            
            % Evaluation
            testCase.verifyEqual(ppfActual11, ppfExpctd11); 
            testCase.verifyEqual(ppfActual21, ppfExpctd21); 
            testCase.verifyEqual(ppfActual12, ppfExpctd12);
            testCase.verifyEqual(ppfActual22, ppfExpctd22);

        end
        
         % Test for plus
        function testPlus(testCase)
            
            % Input value
            coefsA(:,:,1,1,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2,1,1) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,1,2,1) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,2,2,1) = [
                4 4 ;
                4 4 ];
            % Input value
            coefsB(:,:,1,1,1) = [
                1 1 ;
                1 1 ];
            coefsB(:,:,2,1,1) = [
                2 -2 ;
                2 -2 ];
            coefsB(:,:,1,2,1) = [
                3 3 ;
                -3 -3 ];
            coefsB(:,:,2,2,1) = [
                4 -4 ;
                -4 4 ];

            % Expected value
            coefsCExpctd(:,:,1,1,1) = [
                2 2 ;
                2 2 ];
            coefsCExpctd(:,:,2,1,1) = [
                4 0 ;
                4 0 ];
            coefsCExpctd(:,:,1,2,1) = [
                6 6 ;
                0 0 ];
            coefsCExpctd(:,:,2,2,1) = [
                8 0 ;
                0 8 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coefsA);
            testCase.ppm1 = PolyPhaseMatrix3d(coefsB);
            
            % Actual values
            testCase.ppm2 = testCase.ppm0 + testCase.ppm1;
            
            coefsCActual = double(testCase.ppm2);
            
            % Evaluation
            testCase.verifySize(coefsCActual,size(coefsCExpctd));
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
        end
        
        % Test for plus
        function testPlusDifOrd(testCase)
            
            % Input value
            coefsA(:,:,1,1,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2,1,1) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,1,2,1) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,2,2,1) = [
                4 4 ;
                4 4 ];
            % Input value
            coefsB(:,:,1,1,1) = [
                1 1 ;
                1 1 ];
            coefsB(:,:,2,1,1) = [
                2 -2 ;
                2 -2 ];
            coefsB(:,:,1,2,1) = [
                3 3 ;
                -3 -3 ];
            coefsB(:,:,1,2,2) = [
                5 -5 ;
                -5 5 ];

            % Expected value
            coefsCExpctd(:,:,1,1,1) = [
                2 2 ;
                2 2 ];
            coefsCExpctd(:,:,2,1,1) = [
                4 0 ;
                4 0 ];
            coefsCExpctd(:,:,1,2,1) = [
                6 6 ;
                0 0 ];
            coefsCExpctd(:,:,2,2,1) = [
                4 4 ;
                4 4 ];
            coefsCExpctd(:,:,1,2,2) = [
                5 -5 ;
                -5 5 ];   

            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coefsA);
            testCase.ppm1 = PolyPhaseMatrix3d(coefsB);
            
            % Actual values
            testCase.ppm2 = testCase.ppm0 + testCase.ppm1;
            
            coefsCActual = double(testCase.ppm2);
            
            % Evaluation
            testCase.verifySize(coefsCActual,size(coefsCExpctd));
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
        end
        
        % Test for minus
        function testMinus(testCase)
            
            % Input value
            coefsA(:,:,1,1,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2,1,1) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,1,2,1) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,2,2,1) = [
                4 4 ;
                4 4 ];
            
            % Input value
            coefsB(:,:,1,1,1) = [
                1 1 ;
                1 1 ];
            coefsB(:,:,2,1,1) = [
                2 -2 ;
                2 -2 ];
            coefsB(:,:,1,2,1) = [
                3 3 ;
                -3 -3 ];
            coefsB(:,:,1,2,2) = [
                5 -5 ;
                -5 5 ];
            
            % Expected value
            coefsCExpctd(:,:,1,1,1) = [
                0 0 ;
                0 0 ];
            coefsCExpctd(:,:,2,1,1) = [
                0 4 ;
                0 4 ];
            coefsCExpctd(:,:,1,2,1) = [
                0 0 ;
                6 6 ];
            coefsCExpctd(:,:,2,2,1) = [
                4 4 ;
                4 4 ];
            coefsCExpctd(:,:,1,2,2) = [
                -5 5 ;
                5 -5 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coefsA);
            testCase.ppm1 = PolyPhaseMatrix3d(coefsB);
            
            % Actual values
            testCase.ppm2 = testCase.ppm0 - testCase.ppm1;
            
            coefsCActual = double(testCase.ppm2);
            
            % Evaluation
            testCase.verifySize(coefsCActual,size(coefsCExpctd));
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15); 
        end
        
        % Test for mtimes
        function testMTimes(testCase)
            
            % Input value
            coefsA(:,:,1,1,1) = [
                1 0 ;
                0 -1 ];
            coefsA(:,:,2,1,1) = [
                2 0 ;
                0 -2 ];
            coefsA(:,:,1,2,1) = [
                3 0 ;
                0 -3 ];
            coefsA(:,:,2,2,1) = [
                4 0 ;
                0 -4 ];
            coefsA(:,:,1,1,2) = [
                5 0 ;
                0 -5 ];
            coefsA(:,:,2,1,2) = [
                6 0 ;
                0 -6 ];
            coefsA(:,:,1,2,2) = [
                7 0 ;
                0 -7 ];
            coefsA(:,:,2,2,2) = [
                8 0 ;
                0 -8 ];
            
            % Input value
            coefsB(:,:,1,1,1) = [
                1 1 ;
                1 1 ];            
            coefsB(:,:,2,2,2) = [
               -1 -1 ;
               -1 -1 ];
            
            % Expected value
            coefsCExpctd(:,:,1,1,1) = [
                1 1 ;
                -1 -1 ];
            coefsCExpctd(:,:,2,1,1) = [
                2 2 ;
                -2 -2 ];
            coefsCExpctd(:,:,1,2,1) = [
                3 3 ;
                -3 -3 ];
            coefsCExpctd(:,:,2,2,1) = [
                4 4 ;
                -4 -4 ];
            coefsCExpctd(:,:,1,1,2) = [
                5 5 ;
                -5 -5 ];
            coefsCExpctd(:,:,2,1,2) = [
                6 6 ;
                -6 -6 ];
            coefsCExpctd(:,:,1,2,2) = [
                7 7 ;
                -7 -7 ];
            coefsCExpctd(:,:,2,2,2) = [
                7 7 ;
                -7 -7 ];
            coefsCExpctd(:,:,3,2,2) = [
                -2 -2 ;
                2 2 ];
            coefsCExpctd(:,:,2,3,2) = [
                -3 -3 ;
                3 3 ];
            coefsCExpctd(:,:,3,3,2) = [
                -4 -4 ;
                4 4 ];
            coefsCExpctd(:,:,2,2,3) = [
                -5 -5 ;
                5 5 ];
            coefsCExpctd(:,:,3,2,3) = [
                -6 -6 ;
                6 6 ];
            coefsCExpctd(:,:,2,3,3) = [
                -7 -7 ;
                7 7 ];            
            coefsCExpctd(:,:,3,3,3) = [
                -8 -8 ;
                8 8 ];                        
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coefsA);
            testCase.ppm1 = PolyPhaseMatrix3d(coefsB);
            
            % Actual values
            testCase.ppm2 = testCase.ppm0 * testCase.ppm1;
            
            coefsCActual = double(testCase.ppm2);
            
            % Evaluation
            testCase.verifySize(coefsCActual, size(coefsCExpctd));
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
            
        end

        % Test for plus
        function testPlusScalar(testCase)
            
            % Input value
            coefsA(:,:,1,1,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2,1,1) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,1,2,1) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,2,2,1) = [
                4 4 ;
                4 4 ];
            coefsA(:,:,1,1,2) = [
                5 5 ;
                5 5 ];
            coefsA(:,:,2,1,2) = [
                6 6 ;
                6 6 ];
            coefsA(:,:,1,2,2) = [
                7 7 ;
                7 7 ];
            coefsA(:,:,2,2,2) = [
                8 8 ;
                8 8 ];            
            
            % Input value
            scalar = 10;
            
            % Expected value
            coefsCExpctd(:,:,1,1,1) = [
                11    11 ;
                11    11 ];
            coefsCExpctd(:,:,2,1,1) = [
                12    12 ;
                12    12 ];
            coefsCExpctd(:,:,1,2,1) = [
                13    13 ;
                13    13 ];
            coefsCExpctd(:,:,2,2,1) = [
                14    14 ;
                14    14 ];
            coefsCExpctd(:,:,1,1,2) = [
                15    15 ;
                15    15 ];
            coefsCExpctd(:,:,2,1,2) = [
                16    16 ;
                16    16 ];
            coefsCExpctd(:,:,1,2,2) = [
                17    17 ;
                17    17 ];
            coefsCExpctd(:,:,2,2,2) = [
                18    18 ;
                18    18 ];            
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coefsA);
            
            % Actual values
            testCase.ppm1 = testCase.ppm0 + scalar;
            
            coefsCActual = double(testCase.ppm1);
            
            % Evaluation
            testCase.verifySize(coefsCActual, size(coefsCExpctd));
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
            
        end

        % Test for minus scalar
        function testMinusScalar(testCase)
            
            % Input value
            coefsA(:,:,1,1,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2,1,1) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,1,2,1) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,2,2,1) = [
                4 4 ;
                4 4 ];
            coefsA(:,:,1,1,2) = [
                5 5 ;
                5 5 ];
            coefsA(:,:,2,1,2) = [
                6 6 ;
                6 6 ];
            coefsA(:,:,1,2,2) = [
                7 7 ;
                7 7 ];
            coefsA(:,:,2,2,2) = [
                8 8  ;
                8 8 ];            
            
            % Input value
            scalar = 10;
            
            % Expected value
            coefsCExpctd(:,:,1,1,1) = - [
                9   9 ;
                9   9 ];
            coefsCExpctd(:,:,2,1,1) = - [
                8   8 ;
                8   8 ];
            coefsCExpctd(:,:,1,2,1) = - [
                7   7 ;
                7   7 ];
            coefsCExpctd(:,:,2,2,1) = - [
                6   6 ;
                6   6 ];
            coefsCExpctd(:,:,1,1,2) = - [
                5   5 ;
                5   5 ];
            coefsCExpctd(:,:,2,1,2) = - [
                4   4 ;
                4   4 ];
            coefsCExpctd(:,:,1,2,2) = - [
                3   3 ;
                3   3 ];
            coefsCExpctd(:,:,2,2,2) = - [
                2   2 ;
                2   2 ];            
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coefsA);
            
            % Actual valu
            testCase.ppm1 = testCase.ppm0 - scalar;
            
            coefsCActual = double(testCase.ppm1);
            
            % Evaluation
            testCase.verifySize(coefsCActual, size(coefsCExpctd));
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
            
        end
        
        % Test for mtimes
        function testMTimesScalar(testCase)
            
            % Input value
            coefsA(:,:,1,1,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2,1,1) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,1,2,1) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,2,2,1) = [
                4 4 ;
                4 4 ];
            coefsA(:,:,1,1,2) = [
                5 5 ;
                5 5 ];
            coefsA(:,:,2,1,2) = [
                6 6 ;
                6 6 ];
            coefsA(:,:,1,2,2) = [
                7 7 ;
                7 7 ];
            coefsA(:,:,2,2,2) = [
                8 8 ;
                8 8 ];            
            
            % Input value
            scalar = 10;
            
            % Expected value
            coefsCExpctd(:,:,1,1,1) = [
                10 10 ;
                10 10 ];
            coefsCExpctd(:,:,2,1,1) = [
                20 20 ;
                20 20 ];
            coefsCExpctd(:,:,1,2,1) = [
                30 30 ;
                30 30 ];
            coefsCExpctd(:,:,2,2,1) = [
                40 40 ;
                40 40 ];
            coefsCExpctd(:,:,1,1,2) = [
                50 50 ;
                50 50 ];
            coefsCExpctd(:,:,2,1,2) = [
                60 60 ;
                60 60 ];
            coefsCExpctd(:,:,1,2,2) = [
                70 70 ;
                70 70 ];
            coefsCExpctd(:,:,2,2,2) = [
                80 80 ;
                80 80 ];            
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coefsA);
            
            % Actual values
            testCase.ppm1 = testCase.ppm0 * scalar;
            
            coefsCActual = double(testCase.ppm1);
            
            % Evaluation
            testCase.verifySize(coefsCActual, size(coefsCExpctd));
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
            
        end

        % Test for mtimes
        function testMTimes4x4(testCase)
            
            % Input value
            coefsA(:,:,1,1,1) = [
                1  1  1  1;
                1  1 -1 -1;
                1 -1  1 -1;
                1 -1 -1  1 ];
            
            % Input value
            coefsB(:,:,1,1,1) = [
                1  0  0  0;
                0  0  0  0;
                0  0  0  0;
                0  0  0  0 ];
            coefsB(:,:,2,1,1) = [
                0  0  0  0;
                0  1  0  0;
                0  0  0  0;
                0  0  0  0 ];
            coefsB(:,:,1,2,1) = [
                0  0  0  0;
                0  0  0  0;
                0  0  1  0;
                0  0  0  0 ];
            coefsB(:,:,2,2,1) = [
                0  0  0  0;
                0  0  0  0;
                0  0  0  0;
                0  0  0  1];
            coefsB(:,:,1,1,2) = [
                0  0  0  1;
                0  0  0  0;
                0  0  0  0;
                0  0  0  0 ];
            coefsB(:,:,2,1,2) = [
                0  0  0  0;
                0  0  1  0;
                0  0  0  0;
                0  0  0  0 ];
            coefsB(:,:,1,2,2) = [
                0  0  0  0;
                0  0  0  0;
                0  1  0  0;
                0  0  0  0 ];
            coefsB(:,:,2,2,2) = [
                0  0  0  0;
                0  0  0  0;
                0  0  0  0;
                1  0  0  0];
            
            % Expected value
            coefsCExpctd(:,:,1,1,1) = [
                1  0  0  0;
                1  0  0  0;
                1  0  0  0;
                1  0  0  0 ];
            coefsCExpctd(:,:,2,1,1) = [
                0  1  0  0;
                0  1  0  0;
                0 -1  0  0;
                0 -1  0  0 ];
            coefsCExpctd(:,:,1,2,1) = [
                0  0  1  0;
                0  0 -1  0;
                0  0  1  0;
                0  0 -1  0 ];
            coefsCExpctd(:,:,2,2,1) = [
                0  0  0  1;
                0  0  0 -1;
                0  0  0 -1;
                0  0  0  1];
            coefsCExpctd(:,:,1,1,2) = [
                0  0  0  1;
                0  0  0  1;
                0  0  0  1;
                0  0  0  1 ];
            coefsCExpctd(:,:,2,1,2) = [
                0  0  1  0;
                0  0  1  0;
                0  0 -1  0;
                0  0 -1  0 ];
            coefsCExpctd(:,:,1,2,2) = [
                0  1  0 0;
                0 -1  0 0;
                0  1  0 0;
                0 -1  0 0 ];
            coefsCExpctd(:,:,2,2,2) = [
                1  0  0  0;
               -1  0  0  0;
               -1  0  0  0;
                1  0  0  0];            
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coefsA);
            testCase.ppm1 = PolyPhaseMatrix3d(coefsB);
            
            % Actual values
            testCase.ppm2 = testCase.ppm0 * testCase.ppm1;
            
            coefsCActual = double(testCase.ppm2);
            
            % Evaluation
            testCase.verifySize(coefsCActual, size(coefsCExpctd));
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
            
        end

        % Test for ctranspose
        function testCtranspose(testCase)
            
            % Input value
            coef(:,:,1,1,1) = [
                1  0  0  0;
                1i  0  0  0;
                1  0  0  0;
                1i  0  0  0 ];
            coef(:,:,2,1,1) = [
                0  1  0  0;
                0  1i  0  0;
                0 -1  0  0;
                0 -1i  0  0 ];
            coef(:,:,1,2,1) = [
                0  0  1  0;
                0  0 -1i  0;
                0  0  1  0;
                0  0 -1i  0 ];
            coef(:,:,2,2,1) = [
                0  0  0  1;
                0  0  0 -1i;
                0  0  0 -1;
                0  0  0  1i];
            coef(:,:,1,1,2) = [
                1i  0  0  0;
                1  0  0  0;
                1i  0  0  0;
                1  0  0  0 ];
            coef(:,:,2,1,2) = [
                0  1i  0  0;
                0  1  0  0;
                0 -1i  0  0;
                0 -1  0  0 ];
            coef(:,:,1,2,2) = [
                0  0  1i  0;
                0  0 -1  0;
                0  0  1i  0;
                0  0 -1  0 ];
            coef(:,:,2,2,2) = [
                0  0  0  1i;
                0  0  0 -1;
                0  0  0 -1i;
                0  0  0  1 ];            
            
            % Expected value
            coefExpctd(:,:,2,2,2) = [
                1  0  0  0;
                1i  0  0  0;
                1  0  0  0;
                1i  0  0  0 ]';
            coefExpctd(:,:,1,2,2) = [
                0  1  0  0;
                0  1i  0  0;
                0 -1  0  0;
                0 -1i  0  0 ]';
            coefExpctd(:,:,2,1,2) = [
                0  0  1  0;
                0  0 -1i  0;
                0  0  1  0;
                0  0 -1i  0 ]';
            coefExpctd(:,:,1,1,2) = [
                0  0  0  1;
                0  0  0 -1i;
                0  0  0 -1;
                0  0  0  1i]';
            coefExpctd(:,:,2,2,1) = [
                1i  0  0  0;
                1  0  0  0;
                1i  0  0  0;
                1  0  0  0 ]';
            coefExpctd(:,:,1,2,1) = [
                0  1i  0  0;
                0  1  0  0;
                0 -1i  0  0;
                0 -1  0  0 ]';
            coefExpctd(:,:,2,1,1) = [
                0  0  1i  0;
                0  0 -1  0;
                0  0  1i  0;
                0  0 -1  0 ]';
            coefExpctd(:,:,1,1,1) = [
                0  0  0  1i;
                0  0  0 -1;
                0  0  0 -1i;
                0  0  0  1 ]';            
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coef);
            
            % Actual values
            testCase.ppm0 = testCase.ppm0';
            
            coefActual = double(testCase.ppm0);
            
            % Evaluation
            testCase.verifySize(coefActual, size(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
            
        end

        % Test for transpose
        function testTranspose(testCase)
            
            % Input value
            coef(:,:,1,1,1) = [
                1  0  0  0;
                1i  0  0  0;
                1  0  0  0;
                1i  0  0  0 ];
            coef(:,:,2,1,1) = [
                0  1  0  0;
                0  1i  0  0;
                0 -1  0  0;
                0 -1i  0  0 ];
            coef(:,:,1,2,1) = [
                0  0  1  0;
                0  0 -1i  0;
                0  0  1  0;
                0  0 -1i  0 ];
            coef(:,:,2,2,1) = [
                0  0  0  1;
                0  0  0 -1i;
                0  0  0 -1;
                0  0  0  1i];
            coef(:,:,1,1,2) = [
                1i  0  0  0;
                1  0  0  0;
                1i  0  0  0;
                1  0  0  0 ];
            coef(:,:,2,1,2) = [
                0  1i  0  0;
                0  1  0  0;
                0 -1i  0  0;
                0 -1  0  0 ];
            coef(:,:,1,2,2) = [
                0  0  1i  0;
                0  0 -1  0;
                0  0  1i  0;
                0  0 -1  0 ];
            coef(:,:,2,2,2) = [
                0  0  0  1i;
                0  0  0 -1;
                0  0  0 -1i;
                0  0  0  1 ];            
            
            % Expected value
            coefExpctd(:,:,2,2,2) = [
                1  0  0  0;
                1i  0  0  0;
                1  0  0  0;
                1i  0  0  0 ].';
            coefExpctd(:,:,1,2,2) = [
                0  1  0  0;
                0  1i  0  0;
                0 -1  0  0;
                0 -1i  0  0 ].';
            coefExpctd(:,:,2,1,2) = [
                0  0  1  0;
                0  0 -1i  0;
                0  0  1  0;
                0  0 -1i  0 ].';
            coefExpctd(:,:,1,1,2) = [
                0  0  0  1;
                0  0  0 -1i;
                0  0  0 -1;
                0  0  0  1i].';
            coefExpctd(:,:,2,2,1) = [
                1i  0  0  0;
                1  0  0  0;
                1i  0  0  0;
                1  0  0  0 ].';
            coefExpctd(:,:,1,2,1) = [
                0  1i  0  0;
                0  1  0  0;
                0 -1i  0  0;
                0 -1  0  0 ].';
            coefExpctd(:,:,2,1,1) = [
                0  0  1i  0;
                0  0 -1  0;
                0  0  1i  0;
                0  0 -1  0 ].';
            coefExpctd(:,:,1,1,1) = [
                0  0  0  1i;
                0  0  0 -1;
                0  0  0 -1i;
                0  0  0  1 ].';            
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coef);
            
            % Actual values
            testCase.ppm0 = testCase.ppm0.';
            
            coefActual = double(testCase.ppm0);
            
            % Evaluation
            testCase.verifySize(coefActual, size(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
            
        end
        
        % Test for vertical upsampling
        function testUpsampleVertical(testCase)
            
            % Parameters
            factorUy = 2;
            import saivdr.dictionary.utility.*            
            direction = Direction.VERTICAL;
            
            % Input value
            coef(:,:,1,1,1) = [
                1  1 ;
                1  1 ];
            coef(:,:,2,1,1) = [
                2  2 ;
                2  2 ];
            coef(:,:,1,2,1) = [
                3  3 ;
                3  3 ];
            coef(:,:,2,2,1) = [
                4  4 ;
                4  4 ];
            coef(:,:,1,1,2) = [
                5  5 ;
                5  5 ];
            coef(:,:,2,1,2) = [
                6  6 ;
                6  6 ];
            coef(:,:,1,2,2) = [
                7  7 ;
                7  7 ];
            coef(:,:,2,2,2) = [
                8  8 ;
                8  8 ];            
            
            % Expected value
            coefExpctd(:,:,1,1,1) = [
                1  1 ;
                1  1 ];
            coefExpctd(:,:,2,1,1) = [
                0  0 ;
                0  0 ];            
            coefExpctd(:,:,3,1,1) = [
                2  2 ;
                2  2 ];
            coefExpctd(:,:,1,2,1) = [
                3  3 ;
                3  3 ];
            coefExpctd(:,:,2,2,1) = [
                0  0 ;
                0  0 ];            
            coefExpctd(:,:,3,2,1) = [
                4  4 ;
                4  4 ];
            coefExpctd(:,:,1,1,2) = [
                5  5 ;
                5  5 ];
            coefExpctd(:,:,2,1,2) = [
                0  0 ;
                0  0 ];                        
            coefExpctd(:,:,3,1,2) = [
                6  6 ;
                6  6 ];
            coefExpctd(:,:,1,2,2) = [
                7  7 ;
                7  7 ];
            coefExpctd(:,:,2,2,2) = [
                0  0 ;
                0  0 ];                                    
            coefExpctd(:,:,3,2,2) = [
                8  8 ;
                8  8 ];            
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*
            testCase.ppm0 = PolyPhaseMatrix3d(coef);
            
            % Actual value
            coefActual = double(upsample(testCase.ppm0,factorUy,direction));
            
            % Evaluation
            testCase.verifySize(coefActual, size(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
            
        end

        function testUpsampleHorizontal(testCase)
            
            % Parameters
            factorUx = 2;
            import saivdr.dictionary.utility.*
            direction = Direction.HORIZONTAL;
            
            % Input value
            coef(:,:,1,1,1) = [
                1  1 ;
                1  1 ];
            coef(:,:,2,1,1) = [
                2  2 ;
                2  2 ];
            coef(:,:,1,2,1) = [
                3  3 ;
                3  3 ];
            coef(:,:,2,2,1) = [
                4  4 ;
                4  4 ];
            coef(:,:,1,1,2) = [
                5  5 ;
                5  5 ];
            coef(:,:,2,1,2) = [
                6  6 ;
                6  6 ];
            coef(:,:,1,2,2) = [
                7  7 ;
                7  7 ];
            coef(:,:,2,2,2) = [
                8  8 ;
                8  8 ];            
            
            % Expected value
            coefExpctd(:,:,1,1,1) = [
                1  1 ;
                1  1 ];
            coefExpctd(:,:,2,1,1) = [
                2  2 ;
                2  2 ];
            coefExpctd(:,:,1,2,1) = [
                0  0 ;
                0  0 ];            
            coefExpctd(:,:,1,3,1) = [
                3  3 ;
                3  3 ];
            coefExpctd(:,:,2,2,1) = [
                0  0 ;
                0  0 ];            
            coefExpctd(:,:,2,3,1) = [
                4  4 ;
                4  4 ];
            coefExpctd(:,:,1,1,2) = [
                5  5 ;
                5  5 ];
            coefExpctd(:,:,2,1,2) = [
                6  6 ;
                6  6 ];
            coefExpctd(:,:,1,2,2) = [
                0  0 ;
                0  0 ];                        
            coefExpctd(:,:,1,3,2) = [
                7  7 ;
                7  7 ];
            coefExpctd(:,:,2,2,2) = [
                0  0 ;
                0  0 ];                                    
            coefExpctd(:,:,2,3,2) = [
                8  8 ;
                8  8 ];            
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coef);
            
            % Actual value
            coefActual = double(upsample(testCase.ppm0,factorUx,direction));
            
            % Evaluation
            testCase.verifySize(coefActual, size(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
            
        end

        function testUpsampleDepth(testCase)
            
            % Parameters
            factorUz = 2;
            import saivdr.dictionary.utility.*
            direction = Direction.DEPTH;
            
            % Input value
            coef(:,:,1,1,1) = [
                1  1 ;
                1  1 ];
            coef(:,:,2,1,1) = [
                2  2 ;
                2  2 ];
            coef(:,:,1,2,1) = [
                3  3 ;
                3  3 ];
            coef(:,:,2,2,1) = [
                4  4 ;
                4  4 ];
            coef(:,:,1,1,2) = [
                5  5 ;
                5  5 ];
            coef(:,:,2,1,2) = [
                6  6 ;
                6  6 ];
            coef(:,:,1,2,2) = [
                7  7 ;
                7  7 ];
            coef(:,:,2,2,2) = [
                8  8 ;
                8  8 ];            
            
            % Expected value
            coefExpctd(:,:,1,1,1) = [
                1  1 ;
                1  1 ];
            coefExpctd(:,:,2,1,1) = [
                2  2 ;
                2  2 ];
            coefExpctd(:,:,1,2,1) = [
                3  3 ;
                3  3 ];
            coefExpctd(:,:,2,2,1) = [
                4  4 ;
                4  4 ];
            coefExpctd(:,:,1,1,2) = [
                0  0;
                0  0 ];
            coefExpctd(:,:,2,1,2) = [
                0  0;
                0  0 ];
            coefExpctd(:,:,1,2,2) = [
                0  0;
                0  0 ];
            coefExpctd(:,:,2,2,2) = [
                0  0;
                0  0 ];
            coefExpctd(:,:,1,1,3) = [
                5  5 ;
                5  5 ];
            coefExpctd(:,:,2,1,3) = [
                6  6 ;
                6  6 ];
            coefExpctd(:,:,1,2,3) = [
                7  7 ;
                7  7 ];
            coefExpctd(:,:,2,2,3) = [
                8  8 ;
                8  8 ];            
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coef);
            
            % Actual value
            coefActual = double(upsample(testCase.ppm0,factorUz,direction));
            
            % Evaluation
            testCase.verifySize(coefActual, size(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
            
        end
        
        function testUpsampleVerticalHorizontalDepth(testCase)
            
            % Parameters
            factorU = [2 2 2];
            import saivdr.dictionary.utility.*            
            direction = ...
                [ Direction.VERTICAL Direction.HORIZONTAL Direction.DEPTH];
            
            % Input value
            coef(:,:,1,1,1) = [
                1  1 ;
                1  1 ];
            coef(:,:,2,1,1) = [
                2  2 ;
                2  2 ];
            coef(:,:,1,2,1) = [
                3  3 ;
                3  3 ];
            coef(:,:,2,2,1) = [
                4  4 ;
                4  4 ];
            coef(:,:,1,1,2) = [
                5  5 ;
                5  5 ];
            coef(:,:,2,1,2) = [
                6  6 ;
                6  6 ];
            coef(:,:,1,2,2) = [
                7  7 ;
                7  7 ];
            coef(:,:,2,2,2) = [
                8  8 ;
                8  8 ];            
            
            % Expected value
            coefExpctd(:,:,1,1,1) = [
                1  1 ;
                1  1 ];
            coefExpctd(:,:,2,1,1) = [
                0  0 ;
                0  0 ];
            coefExpctd(:,:,3,1,1) = [
                2  2 ;
                2  2 ];
            coefExpctd(:,:,1,2,1) = [
                0  0 ;
                0  0 ];            
            coefExpctd(:,:,2,2,1) = [
                0  0 ;
                0  0 ];                        
            coefExpctd(:,:,3,2,1) = [
                0  0 ;
                0  0 ];                        
            coefExpctd(:,:,1,3,1) = [
                3  3 ;
                3  3 ];
            coefExpctd(:,:,2,3,1) = [
                0  0 ;
                0  0 ];            
            coefExpctd(:,:,3,3,1) = [
                4  4 ;
                4  4 ];
            coefExpctd(:,:,1,1,2) = [
                0  0;
                0  0 ];
            coefExpctd(:,:,2,1,2) = [
                0  0;
                0  0 ];
            coefExpctd(:,:,3,1,2) = [
                0  0;
                0  0 ];            
            coefExpctd(:,:,1,2,2) = [
                0  0;
                0  0 ];
            coefExpctd(:,:,2,2,2) = [
                0  0;
                0  0 ];
            coefExpctd(:,:,3,3,2) = [
                0  0;
                0  0 ];            
            coefExpctd(:,:,1,3,2) = [
                0  0;
                0  0 ];
            coefExpctd(:,:,2,3,2) = [
                0  0;
                0  0 ];
            coefExpctd(:,:,3,3,2) = [
                0  0;
                0  0 ];                        
            coefExpctd(:,:,1,1,3) = [
                5  5 ;
                5  5 ];
            coefExpctd(:,:,2,1,3) = [
                0  0 ;
                0  0 ];            
            coefExpctd(:,:,3,1,3) = [
                6  6 ;
                6  6 ];
            coefExpctd(:,:,1,2,3) = [
                0  0 ;
                0  0 ];            
            coefExpctd(:,:,2,2,3) = [
                0  0 ;
                0  0 ];            
            coefExpctd(:,:,3,2,3) = [
                0  0 ;
                0  0 ];                        
            coefExpctd(:,:,1,3,3) = [
                7  7 ;
                7  7 ];
            coefExpctd(:,:,2,3,3) = [
                0  0 ;
                0  0 ];                        
            coefExpctd(:,:,3,3,3) = [
                8  8 ;
                8  8 ];            
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix3d(coef);
            
            % Actual value
            coefActual = double(upsample(testCase.ppm0,factorU,direction));
            
            % Evaluation
            testCase.verifySize(coefActual, size(coefExpctd));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
            
        end

        function testUpsampleForConstant(testCase)
            
            % Parameters
            factorU = [2 2 2];
            import saivdr.dictionary.utility.*            
            direction = ...
                [ Direction.VERTICAL Direction.HORIZONTAL Direction.DEPTH];
            
            % Input value
            coef = [
                1  1 ;
                1  1 ];
            
            % Expected value
            coefExpctd = [
                1  1 ;
                1  1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*
            testCase.ppm0 = PolyPhaseMatrix3d(coef);
            
            % Actual value
            coefActual = double(upsample(testCase.ppm0,factorU,direction));
            
            % Evaluation
            testCase.verifyEqual(coefExpctd, coefActual);
        end

    end
  
end
