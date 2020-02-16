classdef PolyPhaseCoefOperation1dSystemTestCase < matlab.unittest.TestCase
    %POLYPHASECOEFOPERATION1dTESTCASE Test case for PolyPhaseCoefOperation1dSystem
    %
    % SVN identifier:
    % $Id: PolyPhaseCoefOperation1dSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        ppcos
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.ppcos)
        end
    end
    
    methods (Test)
        
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
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','Plus');
            
            % Actual values
            coefsCActual = step(testCase.ppcos,coefsA,coefsB);
            
            % Evaluation
            testCase.verifySize(coefsCActual,size(coefsCExpctd));
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
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','Minus');
            
            % Actual values
            coefsCActual = step(testCase.ppcos,coefsA,coefsB);
            
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
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','MTimes');
            
            % Actual values
            coefsCActual = step(testCase.ppcos,coefsA,coefsB);
            
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
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','Plus');
            
            % Actual values
            coefsCActual = step(testCase.ppcos,coefsA,scalar);
            
            % Evaluation
            testCase.verifySize(coefsCActual,size(coefsCExpctd));
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
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','Minus');
            
            % Actual value
            coefsCActual = step(testCase.ppcos,coefsA,scalar);
            
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
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','MTimes');
            
            % Actual values
            coefsCActual = step(testCase.ppcos,coefsA,scalar);
            
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
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','MTimes');
            
            % Actual values
            coefsCActual = step(testCase.ppcos,coefsA,coefsB);
            
            % Evaluation
            testCase.verifySize(coefsCActual,size(coefsCExpctd));
            %diff = norm(coefsCExpctd(:)-coefsCActual(:))/numel(coefsCExpctd);
            testCase.verifyEqual(coefsCActual,coefsCExpctd,'RelTol',1e-15);
            
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
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','Char');
            
            % Actual values
            charActual = step(testCase.ppcos,coefs);
            
            % Evaluation
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
            coefs(:,:,4) = [
                0 1 ;
                0 0 ];
            
            % Expected value
            charExpctd = [...
                '[', 10, ... % 10 -> \n
                9, '0,', 9, ... % 9 -> \t
                'z^(-3);', 10,... % 10 -> \n
                9, 'z^(-1),', 9, ... % 9 -> \t
                '0', 10,... % 10 -> \n
                ']'...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','Char');
            
            % Actual values
            charActual = step(testCase.ppcos,coefs);
            
            % Evaluation
            testCase.verifyEqual(charActual, charExpctd);
            
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
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','CTranspose');
            
            % Actual values
            coefActual = step(testCase.ppcos,coef);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
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
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','Transpose');
            
            % Actual values
            coefActual = step(testCase.ppcos,coef);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            %diff = norm(coefExpctd(:)-coefActual(:))/numel(coefExpctd);
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
        end
        
        % Test for upsampling
        function testUpsample(testCase)
            
            % Parameters
            factorU = 2;
            
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
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','Upsample');
            
            % Actual value
            coefActual = step(testCase.ppcos,coef,factorU);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            %diff = norm(coefExpctd(:)-coefActual(:))/numel(coefExpctd);
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
            
        end
        
        function testUpsampleForConstant(testCase)
            
            % Parameters
            factorU = 2;
            
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
            testCase.ppcos = PolyPhaseCoefOperation1dSystem(...
                'Operation','Upsample');
            
            % Actual value
            
            coefActual = step(testCase.ppcos,coef,factorU);
            % Evaluation
            testCase.verifyEqual(coefExpctd, coefActual);
        end
        
    end
end
