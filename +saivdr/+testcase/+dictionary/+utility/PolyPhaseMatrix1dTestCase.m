classdef PolyPhaseMatrix1dTestCase < matlab.unittest.TestCase
    %POLYPHASEMATRIX1dTESTCASE Test case for PolyPhaseMatrix1d
    %
    % SVN identifier:
    % $Id: PolyPhaseMatrix1dTestCase.m 683 2015-05-29 08:22:13Z sho $
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
            testCase.ppm0 = PolyPhaseMatrix1d();
            
            % Actual values
            coefsActual = double(testCase.ppm0);
            
            % Evaluation
            testCase.verifyEqual(coefsActual,coefsExpctd);
            
        end

        % Test for construction with initialization
        function testConstructorWithInit(testCase)
            
            % Input coeffficients
            coefs = [
                1 3
                2 4 ];
            
            % Expected values
            coefsExpctd = coefs;
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coefs);
            
            % Actual values
            coefsActual = double(testCase.ppm0);
            
            % Evaluation
            testCase.verifyEqual(coefsActual,coefsExpctd);
            
        end

        % Test for object construction
        function testConstructorWithObj(testCase)
            
            % Input value
            coefs = [
                1 3 ;
                2 4 ];
            
            % Expected value
            coefsExpctd = coefs;
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coefs);
            testCase.ppm1 = PolyPhaseMatrix1d(testCase.ppm0);
            
            % Actual values
            coefsActual = double(testCase.ppm1);
            
            % Evaluation
            testCase.verifyEqual(coefsActual,coefsExpctd);
            
        end

        % Test for char
        function testChar(testCase)
            
            % Input value
            coefs(:,:,1) = [
                1 0 ;
                0 0 ];
            coefs(:,:,2) = [
                0 0 ;
                2 0 ];
            coefs(:,:,3) = [
                0 3 ;
                0 0 ];
            coefs(:,:,4) = [
                0 0 ;
                0 4 ];
            coefs(:,:,5) = [
                -5 0 ;
                0 0 ];
            coefs(:,:,6) = [
                0 0 ;
                -6 0 ];
            coefs(:,:,7) = [
                0 -7 ;
                0  0 ];
            coefs(:,:,8) = [
                0  0 ;
                0 -8 ];
            
            % Expected value
            charExpctd = [...
                '[', 10, ... % 10 -> \n
                9, '1 - 5*z^(-4),', 9, ... % 9 -> \t
                '3*z^(-2) - 7*z^(-6);', 10,... % 10 -> \n
                9, '2*z^(-1) - 6*z^(-5),', 9, ... % 9 -> \t
                '4*z^(-3) - 8*z^(-7)', 10,... % 10 -> \n
                ']'...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coefs);
            
            % Actual values
            charActual = char(testCase.ppm0);
            
            % Display
            %             disp(testCase.ppm0);
            %             x = 1;
            %             y = 1;
            %             eval(char(testCase.ppm0));
            
            % Evaluation
            %strcmp(charExpctd,charActual)
            testCase.verifyEqual(charActual, charExpctd);
            
        end
        
        % Test for char with zero elements
        function testCharWithZeros(testCase)
            
            % Input value
            coefs(:,:,1) = [
                0 0 ;
                0 0 ];
            coefs(:,:,2) = [
                0 0 ;
                1 0 ];
            coefs(:,:,3) = [
                0 1 ;
                0 0 ];
            
            % Expected value
            charExpctd = [...
                '[', 10, ... % 10 -> \n
                9, '0,', 9, ... % 9 -> \t
                'z^(-2);', 10,... % 10 -> \n
                9, 'z^(-1),', 9, ... % 9 -> \t
                '0', 10,... % 10 -> \n
                ']'...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coefs);
            
            % Actual values
            charActual = char(testCase.ppm0);
            
            % Display
            %             display(charExpctd);
            %             display(charActual);
            %             x = 2;
            %             y = 2;
            %             eval(char(testCase.ppm0));
            
            % Evaluation
            %strcmp(charExpctd,charActual)
            testCase.verifyEqual(charActual, charExpctd);
            
        end

        % Test for subsref
        function testSubsRef(testCase)
            
            % Input value
            coefs(:,:,1) = [
                1 1 ;
                1 1 ];
            coefs(:,:,2) = [
                2 -2 ;
                2 -2 ];
            coefs(:,:,3) = [
                3 3 ;
                -3 -3 ];
            coefs(:,:,4) = [
                4 -4 ;
                -4 4 ];
            
            % Expected value
            ppfExpctd11 = [ 1  2  3  4 ];
            ppfExpctd21 = [ 1  2 -3 -4 ];
            ppfExpctd12 = [ 1 -2  3 -4 ];
            ppfExpctd22 = [ 1 -2 -3  4 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coefs);
            
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
            coefsA(:,:,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,3) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,4) = [
                4 4 ;
                4 4 ];
            
            % Input value
            coefsB(:,:,1) = [
                1 1 ;
                1 1 ];
            coefsB(:,:,2) = [
                2 -2 ;
                2 -2 ];
            coefsB(:,:,3) = [
                3 3 ;
                -3 -3 ];
            coefsB(:,:,5) = [
                5 -5 ;
                -5 5 ];
            
            % Expected value
            coefsCExpctd(:,:,1) = [
                2 2 ;
                2 2 ];
            coefsCExpctd(:,:,2) = [
                4 0 ;
                4 0 ];
            coefsCExpctd(:,:,3) = [
                6 6 ;
                0 0 ];
            coefsCExpctd(:,:,4) = [
                4 4 ;
                4 4 ];
            coefsCExpctd(:,:,5) = [
                5 -5 ;
                -5 5 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coefsA);
            testCase.ppm1 = PolyPhaseMatrix1d(coefsB);
            
            % Actual values
            testCase.ppm2 = testCase.ppm0 + testCase.ppm1;
            
            coefsCActual = double(testCase.ppm2);
            
            % Evaluation
            testCase.verifySize(coefsCActual, size(coefsCExpctd));
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
        end

        % Test for minus
        function testMinus(testCase)
            
            % Input value
            coefsA(:,:,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,3) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,4) = [
                4 4 ;
                4 4 ];
            
            % Input value
            coefsB(:,:,1) = [
                1 1 ;
                1 1 ];
            coefsB(:,:,2) = [
                2 -2 ;
                2 -2 ];
            coefsB(:,:,3) = [
                3 3 ;
                -3 -3 ];
            coefsB(:,:,5) = [
                5 -5 ;
                -5 5 ];
            
            % Expected value
            coefsCExpctd(:,:,1) = [
                0 0 ;
                0 0 ];
            coefsCExpctd(:,:,2) = [
                0 4 ;
                0 4 ];
            coefsCExpctd(:,:,3) = [
                0 0 ;
                6 6 ];
            coefsCExpctd(:,:,4) = [
                4 4 ;
                4 4 ];
            coefsCExpctd(:,:,5) = [
                -5 5 ;
                5 -5 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coefsA);
            testCase.ppm1 = PolyPhaseMatrix1d(coefsB);
            
            % Actual values
            testCase.ppm2 = testCase.ppm0 - testCase.ppm1;
            
            coefsCActual = double(testCase.ppm2);
            
            % Evaluation
            testCase.verifySize(coefsCActual,size(coefsCExpctd));
            %diff = norm(coefsCExpctd(:)-coefsCActual(:))/numel(coefsCExpctd);
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15); 
        end

        % Test for mtimes
        function testMTimes(testCase)
            
            % Input value
            coefsA(:,:,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,3) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,4) = [
                4 4 ;
                4 4 ];
            
            % Input value
            coefsB(:,:,1) = [
                1 1 ;
                1 1 ];
            coefsB(:,:,2) = [
                1 -1 ;
                1 -1 ];
            coefsB(:,:,3) = [
                1 1 ;
                -1 -1 ];
            coefsB(:,:,4) = [
                1 -1 ;
                -1  1 ];
            
            % Expected value
            coefsCExpctd(:,:,1) = [
                2     2 ;
                2     2 ];
            coefsCExpctd(:,:,2) = [
                6     2 ;
                6     2 ];
            coefsCExpctd(:,:,3) = [
               10     2 ;
               10     2 ];
            coefsCExpctd(:,:,4) = [
               14     2 ;
               14     2 ];
            coefsCExpctd(:,:,5) = [
                8    -8 ;
                8    -8 ];
            coefsCExpctd(:,:,6) = [
                0     0 ;
                0     0 ];
            coefsCExpctd(:,:,7) = [
                0     0 ;
                0     0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coefsA);
            testCase.ppm1 = PolyPhaseMatrix1d(coefsB);
            
            % Actual values
            testCase.ppm2 = testCase.ppm0 * testCase.ppm1;
            
            coefsCActual = double(testCase.ppm2);
            
            % Evaluation
            testCase.verifySize(coefsCActual,size(coefsCExpctd));
            %diff = norm(coefsCExpctd(:)-coefsCActual(:))/numel(coefsCExpctd);
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
            
        end

        % Test for mtimes
        function testPlusScalar(testCase)
            
            % Input value
            coefsA(:,:,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,3) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,4) = [
                4 4 ;
                4 4 ];
            
            % Input value
            scalar = 10;
            
            % Expected value
            coefsCExpctd(:,:,1) = [
                11    11 ;
                11    11 ];
            coefsCExpctd(:,:,2) = [
                12    12 ;
                12     12 ];
            coefsCExpctd(:,:,3) = [
                13    13 ;
                13    13 ];
            coefsCExpctd(:,:,4) = [
                14    14 ;
                14    14 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coefsA);
            
            % Actual values
            testCase.ppm1 = testCase.ppm0 + scalar;
            
            coefsCActual = double(testCase.ppm1);
            
            % Evaluation
            testCase.verifySize(coefsCActual, size(coefsCExpctd));
            %diff = norm(coefsCExpctd(:)-coefsCActual(:))/numel(coefsCExpctd);
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
            
        end

        % Test for minus scalar
        function testMinusScalar(testCase)
            
            % Input value
            coefsA(:,:,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,3) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,4) = [
                4 4 ;
                4 4 ];
            
            % Input value
            scalar = 10;
            
            % Expected value
            coefsCExpctd(:,:,1) = - [
                9   9 ;
                9   9 ];
            coefsCExpctd(:,:,2) = - [
                8   8 ;
                8   8 ];
            coefsCExpctd(:,:,3) = - [
                7   7 ;
                7   7 ];
            coefsCExpctd(:,:,4) = - [
                6   6 ;
                6   6 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coefsA);
            
            % Actual valu
            testCase.ppm1 = testCase.ppm0 - scalar;
            
            coefsCActual = double(testCase.ppm1);
            
            % Evaluation
            testCase.verifySize(coefsCActual,size(coefsCExpctd));
            %diff = norm(coefsCExpctd(:)-coefsCActual(:))/numel(coefsCExpctd);
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
            
        end

        % Test for mtimes
        function testMTimesScalar(testCase)
            
            % Input value
            coefsA(:,:,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,3) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,4) = [
                4 4 ;
                4 4 ];
            
            % Input value
            scalar = 10;
            
            % Expected value
            coefsCExpctd(:,:,1) = [
                10 10 ;
                10 10 ];
            coefsCExpctd(:,:,2) = [
                20 20 ;
                20 20 ];
            coefsCExpctd(:,:,3) = [
                30 30 ;
                30 30 ];
            coefsCExpctd(:,:,4) = [
                40 40 ;
                40 40 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coefsA);
            
            % Actual values
            testCase.ppm1 = testCase.ppm0 * scalar;
            
            coefsCActual = double(testCase.ppm1);
            
            % Evaluation
            testCase.verifySize(coefsCActual,size(coefsCExpctd));
            %diff = norm(coefsCExpctd(:)-coefsCActual(:))/numel(coefsCExpctd);
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
            
        end

        % Test for mtimes
        function testMTimes4x4(testCase)
            
            % Input value
            coefsA(:,:,1) = [
                1  1  1  1;
                1  1 -1 -1;
                1 -1  1 -1;
                1 -1 -1  1 ];
            
            % Input value
            coefsB(:,:,1) = [
                1  0  0  0;
                0  0  0  0;
                0  0  0  0;
                0  0  0  0 ];
            coefsB(:,:,2) = [
                0  0  0  0;
                0  1  0  0;
                0  0  0  0;
                0  0  0  0 ];
            coefsB(:,:,3) = [
                0  0  0  0;
                0  0  0  0;
                0  0  1  0;
                0  0  0  0 ];
            coefsB(:,:,4) = [
                0  0  0  0;
                0  0  0  0;
                0  0  0  0;
                0  0  0  1];
            
            % Expected value
            coefsCExpctd(:,:,1) = [
                1  0  0  0;
                1  0  0  0;
                1  0  0  0;
                1  0  0  0 ];
            coefsCExpctd(:,:,2) = [
                0  1  0  0;
                0  1  0  0;
                0 -1  0  0;
                0 -1  0  0 ];
            coefsCExpctd(:,:,3) = [
                0  0  1  0;
                0  0 -1  0;
                0  0  1  0;
                0  0 -1  0 ];
            coefsCExpctd(:,:,4) = [
                0  0  0  1;
                0  0  0 -1;
                0  0  0 -1;
                0  0  0  1];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coefsA);
            testCase.ppm1 = PolyPhaseMatrix1d(coefsB);
            
            % Actual values
            testCase.ppm2 = testCase.ppm0 * testCase.ppm1;
            
            coefsCActual = double(testCase.ppm2);
            
            % Evaluation
            testCase.verifySize(coefsCActual,size(coefsCExpctd));
            %diff = norm(coefsCExpctd(:)-coefsCActual(:))/numel(coefsCExpctd);
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
            
        end

        % Test for ctranspose
        function testCtranspose(testCase)
            
            % Input value
            coef(:,:,1) = [
                1  0  0  0;
                1i  0  0  0;
                1  0  0  0;
                1i  0  0  0 ];
            coef(:,:,2) = [
                0  1  0  0;
                0  1i  0  0;
                0 -1  0  0;
                0 -1i  0  0 ];
            coef(:,:,3) = [
                0  0  1  0;
                0  0 -1i  0;
                0  0  1  0;
                0  0 -1i  0 ];
            coef(:,:,4) = [
                0  0  0  1;
                0  0  0 -1i;
                0  0  0 -1;
                0  0  0  1i];
            
            % Expected value
            coefExpctd(:,:,4) = [
                1 -1i  1 -1i;
                0  0  0  0;
                0  0  0  0;
                0  0  0  0 ];
            coefExpctd(:,:,3) = [
                0  0  0  0;
                1 -1i -1  1i;
                0  0  0  0;
                0  0  0  0 ];
            coefExpctd(:,:,2) = [
                0  0  0  0;
                0  0  0  0;
                1  1i  1  1i;
                0  0  0  0 ];
            coefExpctd(:,:,1) = [
                0  0  0  0;
                0  0  0  0;
                0  0  0  0;
                1  1i -1 -1i];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coef);
            
            % Actual values
            testCase.ppm0 = testCase.ppm0';
            
            coefActual = double(testCase.ppm0);
            
            % Evaluation
            testCase.verifySize(coefActual, size(coefExpctd));
            %diff = norm(coefExpctd(:)-coefActual(:))/numel(coefExpctd);
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
            
        end

        % Test for transpose
        function testTranspose(testCase)
            
            % Input value
            coef(:,:,1) = [
                1  0  0  0;
                1i  0  0  0;
                1  0  0  0;
                1i  0  0  0 ];
            coef(:,:,2) = [
                0  1  0  0;
                0  1i  0  0;
                0 -1  0  0;
                0 -1i  0  0 ];
            coef(:,:,3) = [
                0  0  1  0;
                0  0 -1i  0;
                0  0  1  0;
                0  0 -1i  0 ];
            coef(:,:,4) = [
                0  0  0  1;
                0  0  0 -1i;
                0  0  0 -1;
                0  0  0  1i];
            
            % Expected value
            coefExpctd(:,:,4) = [
                1  1i  1  1i;
                0  0  0  0;
                0  0  0  0;
                0  0  0  0 ];
            coefExpctd(:,:,3) = [
                0  0  0  0;
                1  1i -1 -1i;
                0  0  0  0;
                0  0  0  0 ];
            coefExpctd(:,:,2) = [
                0  0  0  0;
                0  0  0  0;
                1 -1i  1 -1i;
                0  0  0  0 ];
            coefExpctd(:,:,1) = [
                0  0  0  0;
                0  0  0  0;
                0  0  0  0;
                1 -1i -1  1i];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppm0 = PolyPhaseMatrix1d(coef);
            
            % Actual values
            testCase.ppm0 = testCase.ppm0.';
            
            coefActual = double(testCase.ppm0);
            
            % Evaluation
            testCase.verifySize(coefActual, size(coefExpctd));
            %diff = norm(coefExpctd(:)-coefActual(:))/numel(coefExpctd);
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
        end
        
        % Test for upsampling
        function testUpsample(testCase)
            
            % Parameters
            factorU = 2;
            import saivdr.dictionary.utility.*            
            
            % Input value
            coef(:,:,1) = [
                1  1 ;
                1  1 ];
            coef(:,:,2) = [
                2  2 ;
                2  2 ];
            coef(:,:,3) = [
                3  3 ;
                3  3 ];
            coef(:,:,4) = [
                4  4 ;
                4  4 ];
            
            % Expected value
            coefExpctd(:,:,1) = [
                1  1 ;
                1  1 ];
            coefExpctd(:,:,2) = [
                0  0 ;
                0  0 ];
            coefExpctd(:,:,3) = [
                2  2 ;
                2  2 ];
            coefExpctd(:,:,4) = [
                0  0 ;
                0  0 ];
            coefExpctd(:,:,5) = [
                3  3 ;
                3  3 ];
            coefExpctd(:,:,6) = [
                0  0 ;
                0  0 ];            
            coefExpctd(:,:,7) = [
                4  4 ;
                4  4 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*
            testCase.ppm0 = PolyPhaseMatrix1d(coef);
            
            % Actual value
            coefActual = double(upsample(testCase.ppm0,factorU));
            
            % Evaluation
            testCase.verifySize(coefActual, size(coefExpctd));
            %diff = norm(coefExpctd(:)-coefActual(:))/numel(coefExpctd);
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
            
        end

        function testUpsampleForConstant(testCase)
            
            % Parameters
            factorU = 2;
            import saivdr.dictionary.utility.*            
            
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
            testCase.ppm0 = PolyPhaseMatrix1d(coef);
            
            % Actual value
            coefActual = double(upsample(testCase.ppm0,factorU));
            
            % Evaluation
            testCase.verifyEqual(coefActual, coefExpctd);
        end
    
    end
end
