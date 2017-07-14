classdef PolyPhaseCoefOperation2dSystemTestCase < matlab.unittest.TestCase
    %POLYPHASECOEFOPERATION2DTESTCASE Test case for PolyPhaseCoefOperation2dSystem
    %
    % SVN identifier:
    % $Id: PolyPhaseCoefOperation2dSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
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
            coefsA(:,:,1,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2,1) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,1,2) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,2,2) = [
                4 4 ;
                4 4 ];
            
            % Input value
            coefsB(:,:,1,1) = [
                1 1 ;
                1 1 ];
            coefsB(:,:,2,1) = [
                2 -2 ;
                2 -2 ];
            coefsB(:,:,1,2) = [
                3 3 ;
                -3 -3 ];
            coefsB(:,:,3,3) = [
                5 -5 ;
                -5 5 ];
            
            % Expected value
            coefsCExpctd(:,:,1,1) = [
                2 2 ;
                2 2 ];
            coefsCExpctd(:,:,2,1) = [
                4 0 ;
                4 0 ];
            coefsCExpctd(:,:,1,2) = [
                6 6 ;
                0 0 ];
            coefsCExpctd(:,:,2,2) = [
                4 4 ;
                4 4 ];
            coefsCExpctd(:,:,3,3) = [
                5 -5 ;
                -5 5 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
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
            coefsA(:,:,1,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2,1) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,1,2) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,2,2) = [
                4 4 ;
                4 4 ];
            
            % Input value
            coefsB(:,:,1,1) = [
                1 1 ;
                1 1 ];
            coefsB(:,:,2,1) = [
                2 -2 ;
                2 -2 ];
            coefsB(:,:,1,2) = [
                3 3 ;
                -3 -3 ];
            coefsB(:,:,3,3) = [
                5 -5 ;
                -5 5 ];
            
            % Expected value
            coefsCExpctd(:,:,1,1) = [
                0 0 ;
                0 0 ];
            coefsCExpctd(:,:,2,1) = [
                0 4 ;
                0 4 ];
            coefsCExpctd(:,:,1,2) = [
                0 0 ;
                6 6 ];
            coefsCExpctd(:,:,2,2) = [
                4 4 ;
                4 4 ];
            coefsCExpctd(:,:,3,3) = [
                -5 5 ;
                5 -5 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
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
            coefsA(:,:,1,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2,1) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,1,2) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,2,2) = [
                4 4 ;
                4 4 ];
            
            % Input value
            coefsB(:,:,1,1) = [
                1 1 ;
                1 1 ];
            coefsB(:,:,2,1) = [
                1 -1 ;
                1 -1 ];
            coefsB(:,:,1,2) = [
                1 1 ;
                -1 -1 ];
            coefsB(:,:,2,2) = [
                1 -1 ;
                -1  1 ];
            
            % Expected value
            coefsCExpctd(:,:,1,1) = [
                2     2 ;
                2     2 ];
            coefsCExpctd(:,:,2,1) = [
                6     2 ;
                6     2 ];
            coefsCExpctd(:,:,3,1) = [
                4    -4 ;
                4    -4 ];
            coefsCExpctd(:,:,1,2) = [
                6     6 ;
                6     6 ];
            coefsCExpctd(:,:,2,2) = [
                14     2 ;
                14     2 ];
            coefsCExpctd(:,:,3,2) = [
                8    -8 ;
                8    -8 ];
            coefsCExpctd(:,:,1,3) = [
                0     0 ;
                0     0 ];
            coefsCExpctd(:,:,2,3) = [
                0     0
                0     0 ];
            coefsCExpctd(:,:,3,3) = [
                0     0
                0     0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
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
            coefsA(:,:,1,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2,1) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,1,2) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,2,2) = [
                4 4 ;
                4 4 ];
            
            % Input value
            scalar = 10;
            
            % Expected value
            coefsCExpctd(:,:,1,1) = [
                11    11 ;
                11    11 ];
            coefsCExpctd(:,:,2,1) = [
                12    12 ;
                12     12 ];
            coefsCExpctd(:,:,1,2) = [
                13    13 ;
                13    13 ];
            coefsCExpctd(:,:,2,2) = [
                14    14 ;
                14    14 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
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
            coefsA(:,:,1,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2,1) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,1,2) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,2,2) = [
                4 4 ;
                4 4 ];
            
            % Input value
            scalar = 10;
            
            % Expected value
            coefsCExpctd(:,:,1,1) = - [
                9   9 ;
                9   9 ];
            coefsCExpctd(:,:,2,1) = - [
                8   8 ;
                8   8 ];
            coefsCExpctd(:,:,1,2) = - [
                7   7 ;
                7   7 ];
            coefsCExpctd(:,:,2,2) = - [
                6   6 ;
                6   6 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
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
            coefsA(:,:,1,1) = [
                1 1 ;
                1 1 ];
            coefsA(:,:,2,1) = [
                2 2 ;
                2 2 ];
            coefsA(:,:,1,2) = [
                3 3 ;
                3 3 ];
            coefsA(:,:,2,2) = [
                4 4 ;
                4 4 ];
            
            % Input value
            scalar = 10;
            
            % Expected value
            coefsCExpctd(:,:,1,1) = [
                10 10 ;
                10 10 ];
            coefsCExpctd(:,:,2,1) = [
                20 20 ;
                20 20 ];
            coefsCExpctd(:,:,1,2) = [
                30 30 ;
                30 30 ];
            coefsCExpctd(:,:,2,2) = [
                40 40 ;
                40 40 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
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
            coefsA(:,:,1,1) = [
                1  1  1  1;
                1  1 -1 -1;
                1 -1  1 -1;
                1 -1 -1  1 ];
            
            % Input value
            coefsB(:,:,1,1) = [
                1  0  0  0;
                0  0  0  0;
                0  0  0  0;
                0  0  0  0 ];
            coefsB(:,:,2,1) = [
                0  0  0  0;
                0  1  0  0;
                0  0  0  0;
                0  0  0  0 ];
            coefsB(:,:,1,2) = [
                0  0  0  0;
                0  0  0  0;
                0  0  1  0;
                0  0  0  0 ];
            coefsB(:,:,2,2) = [
                0  0  0  0;
                0  0  0  0;
                0  0  0  0;
                0  0  0  1];
            
            % Expected value
            coefsCExpctd(:,:,1,1) = [
                1  0  0  0;
                1  0  0  0;
                1  0  0  0;
                1  0  0  0 ];
            coefsCExpctd(:,:,2,1) = [
                0  1  0  0;
                0  1  0  0;
                0 -1  0  0;
                0 -1  0  0 ];
            coefsCExpctd(:,:,1,2) = [
                0  0  1  0;
                0  0 -1  0;
                0  0  1  0;
                0  0 -1  0 ];
            coefsCExpctd(:,:,2,2) = [
                0  0  0  1;
                0  0  0 -1;
                0  0  0 -1;
                0  0  0  1];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
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
            coefs(:,:,1,1) = [
                1 0 ;
                0 0 ];
            coefs(:,:,2,1) = [
                0 0 ;
                2 0 ];
            coefs(:,:,1,2) = [
                0 3 ;
                0 0 ];
            coefs(:,:,2,2) = [
                0 0 ;
                0 4 ];
            coefs(:,:,3,1) = [
                -5 0 ;
                0 0 ];
            coefs(:,:,1,3) = [
                0 0 ;
                -6 0 ];
            coefs(:,:,3,2) = [
                0 -7 ;
                0  0 ];
            coefs(:,:,2,3) = [
                0  0 ;
                0 -8 ];
            
            % Expected value
            charExpctd = [...
                '[', 10, ... % 10 -> \n
                9, '1 - 5*y^(-2),', 9, ... % 9 -> \t
                '3*x^(-1) - 7*y^(-2)*x^(-1);', 10,... % 10 -> \n
                9, '2*y^(-1) - 6*x^(-2),', 9, ... % 9 -> \t
                '4*y^(-1)*x^(-1) - 8*y^(-1)*x^(-2)', 10,... % 10 -> \n
                ']'...
                ];
            
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
                'Operation','Char');
            
            % Actual values
            charActual = step(testCase.ppcos,coefs);
            
            % Evaluation
            testCase.verifyEqual(charExpctd, charActual);
            
        end
        
        % Test for char with zero elements
        function testCharWithZeros(testCase)
            
            % Input value
            coefs(:,:,1,1) = [
                0 0 ;
                0 0 ];
            coefs(:,:,2,1) = [
                0 0 ;
                1 0 ];
            coefs(:,:,1,2) = [
                0 1 ;
                0 0 ];
            
            % Expected value
            charExpctd = [...
                '[', 10, ... % 10 -> \n
                9, '0,', 9, ... % 9 -> \t
                'x^(-1);', 10,... % 10 -> \n
                9, 'y^(-1),', 9, ... % 9 -> \t
                '0', 10,... % 10 -> \n
                ']'...
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
                'Operation','Char');
            
            % Actual values
            charActual = step(testCase.ppcos,coefs);
            
            % Evaluation
            testCase.verifyEqual(charExpctd, charActual);
            
        end
        
        % Test for ctranspose
        function testCtranspose(testCase)
            
            % Input value
            coef(:,:,1,1) = [
                1  0  0  0;
                1i  0  0  0;
                1  0  0  0;
                1i  0  0  0 ];
            coef(:,:,2,1) = [
                0  1  0  0;
                0  1i  0  0;
                0 -1  0  0;
                0 -1i  0  0 ];
            coef(:,:,1,2) = [
                0  0  1  0;
                0  0 -1i  0;
                0  0  1  0;
                0  0 -1i  0 ];
            coef(:,:,2,2) = [
                0  0  0  1;
                0  0  0 -1i;
                0  0  0 -1;
                0  0  0  1i];
            
            % Expected value
            coefExpctd(:,:,2,2) = [
                1 -1i  1 -1i;
                0  0  0  0;
                0  0  0  0;
                0  0  0  0 ];
            coefExpctd(:,:,1,2) = [
                0  0  0  0;
                1 -1i -1  1i;
                0  0  0  0;
                0  0  0  0 ];
            coefExpctd(:,:,2,1) = [
                0  0  0  0;
                0  0  0  0;
                1  1i  1  1i;
                0  0  0  0 ];
            coefExpctd(:,:,1,1) = [
                0  0  0  0;
                0  0  0  0;
                0  0  0  0;
                1  1i -1 -1i];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
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
            coef(:,:,1,1) = [
                1  0  0  0;
                1i  0  0  0;
                1  0  0  0;
                1i  0  0  0 ];
            coef(:,:,2,1) = [
                0  1  0  0;
                0  1i  0  0;
                0 -1  0  0;
                0 -1i  0  0 ];
            coef(:,:,1,2) = [
                0  0  1  0;
                0  0 -1i  0;
                0  0  1  0;
                0  0 -1i  0 ];
            coef(:,:,2,2) = [
                0  0  0  1;
                0  0  0 -1i;
                0  0  0 -1;
                0  0  0  1i];
            
            % Expected value
            coefExpctd(:,:,2,2) = [
                1  1i  1  1i;
                0  0  0  0;
                0  0  0  0;
                0  0  0  0 ];
            coefExpctd(:,:,1,2) = [
                0  0  0  0;
                1  1i -1 -1i;
                0  0  0  0;
                0  0  0  0 ];
            coefExpctd(:,:,2,1) = [
                0  0  0  0;
                0  0  0  0;
                1 -1i  1 -1i;
                0  0  0  0 ];
            coefExpctd(:,:,1,1) = [
                0  0  0  0;
                0  0  0  0;
                0  0  0  0;
                1 -1i -1  1i];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
                'Operation','Transpose');
            
            % Actual values
            coefActual = step(testCase.ppcos,coef);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            %diff = norm(coefExpctd(:)-coefActual(:))/numel(coefExpctd);
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
        end
        
        % Test for vertical upsampling
        function testUpsampleVertical(testCase)
            
            % Parameters
            factorUy = 2;
            import saivdr.dictionary.utility.Direction
            direction = Direction.VERTICAL;
            
            % Input value
            coef(:,:,1,1) = [
                1  1 ;
                1  1 ];
            coef(:,:,2,1) = [
                2  2 ;
                2  2 ];
            coef(:,:,1,2) = [
                3  3 ;
                3  3 ];
            coef(:,:,2,2) = [
                4  4 ;
                4  4 ];
            
            % Expected value
            coefExpctd(:,:,1,1) = [
                1  1 ;
                1  1 ];
            coefExpctd(:,:,2,1) = [
                0  0 ;
                0  0 ];
            coefExpctd(:,:,3,1) = [
                2  2 ;
                2  2 ];
            coefExpctd(:,:,1,2) = [
                3  3 ;
                3  3 ];
            coefExpctd(:,:,2,2) = [
                0  0 ;
                0  0 ];
            coefExpctd(:,:,3,2) = [
                4  4 ;
                4  4 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
                'Operation','Upsample');
            
            % Actual value
            coefActual = step(testCase.ppcos,coef,factorUy,direction);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            %diff = norm(coefExpctd(:)-coefActual(:))/numel(coefExpctd);
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
            
        end
        
        function testUpsampleHorizontal(testCase)
            
            % Parameters
            factorUy = 2;
            import saivdr.dictionary.utility.*
            direction = Direction.HORIZONTAL;
            
            % Input value
            coef(:,:,1,1) = [
                1  1 ;
                1  1 ];
            coef(:,:,2,1) = [
                2  2 ;
                2  2 ];
            coef(:,:,1,2) = [
                3  3 ;
                3  3 ];
            coef(:,:,2,2) = [
                4  4 ;
                4  4 ];
            
            % Expected value
            coefExpctd(:,:,1,1) = [
                1  1 ;
                1  1 ];
            coefExpctd(:,:,2,1) = [
                2  2 ;
                2  2 ];
            coefExpctd(:,:,1,2) = [
                0  0 ;
                0  0 ];
            coefExpctd(:,:,1,3) = [
                3  3 ;
                3  3 ];
            coefExpctd(:,:,2,2) = [
                0  0 ;
                0  0 ];
            coefExpctd(:,:,2,3) = [
                4  4 ;
                4  4 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            import saivdr.dictionary.utility.PolyPhaseCoefOperation2dSystem
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
                'Operation','Upsample');
            
            % Actual value
            coefActual = step(testCase.ppcos,coef,factorUy,direction);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            %diff = norm(coefExpctd(:)-coefActual(:))/numel(coefExpctd);
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
            
        end
        
        function testUpsampleVerticalHorizontal(testCase)
            
            % Parameters
            factorUy = [2 2];
            import saivdr.dictionary.utility.*            
            direction = ...
                [ Direction.VERTICAL Direction.HORIZONTAL ];
            
            % Input value
            coef(:,:,1,1) = [
                1  1 ;
                1  1 ];
            coef(:,:,2,1) = [
                2  2 ;
                2  2 ];
            coef(:,:,1,2) = [
                3  3 ;
                3  3 ];
            coef(:,:,2,2) = [
                4  4 ;
                4  4 ];
            
            % Expected value
            coefExpctd(:,:,1,1) = [
                1  1 ;
                1  1 ];
            coefExpctd(:,:,2,1) = [
                0  0 ;
                0  0 ];
            coefExpctd(:,:,3,1) = [
                2  2 ;
                2  2 ];
            coefExpctd(:,:,1,2) = [
                0  0 ;
                0  0 ];
            coefExpctd(:,:,2,2) = [
                0  0 ;
                0  0 ];
            coefExpctd(:,:,2,3) = [
                0  0 ;
                0  0 ];
            coefExpctd(:,:,1,3) = [
                3  3 ;
                3  3 ];
            coefExpctd(:,:,2,3) = [
                0  0 ;
                0  0 ];
            coefExpctd(:,:,3,3) = [
                4  4 ;
                4  4 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
                'Operation','Upsample');
            
            % Actual value
            coefActual = step(testCase.ppcos,coef,factorUy,direction);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            %diff = norm(coefExpctd(:)-coefActual(:))/numel(coefExpctd);
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-15);
            
        end
        
        function testUpsampleForConstant(testCase)
            
            % Parameters
            factorUy = [2 2];
            direction = ...
                [ Direction.VERTICAL Direction.HORIZONTAL ];
            
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
            testCase.ppcos = PolyPhaseCoefOperation2dSystem(...
                'Operation','Upsample');
            
            % Actual value

            coefActual = step(testCase.ppcos,coef,factorUy,direction);            
            % Evaluation
            testCase.verifyEqual(coefExpctd, coefActual);
        end

    end
end
