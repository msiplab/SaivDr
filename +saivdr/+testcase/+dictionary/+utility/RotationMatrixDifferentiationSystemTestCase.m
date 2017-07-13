classdef RotationMatrixDifferentiationSystemTestCase < matlab.unittest.TestCase
    %ROTATIONMATRIXDIFFERENTIATIONSYSTEMTESTCASE Test case for RotationMatrixDifferentiationSystem
    %
    % SVN identifier:
    % $Id: RotationMatrixDifferentiationSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        rmds
    end
    
    methods (TestMethodTeardown) 
        function deleteObject(testCase)
            delete(testCase.rmds);
        end
    end
    
    methods (Test)

        % Test for default construction
        function testConstructor(testCase)
            
            % Expected values
            coefExpctd = [
                0 -1 ;
                1  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*
            testCase.rmds =  RotationMatrixDifferentiationSystem();
            
            % Actual values
            coefActual = step(testCase.rmds,0,1,1);
            
            % Evaluation
            testCase.verifyEqual(coefExpctd,coefActual,'AbsTol',1e-10);
        end

        % Test for default construction
        function testConstructorWithAngles(testCase)
            
            % Expected values
            coefExpctd = [
                -sin(pi/4) -cos(pi/4) ;
                 cos(pi/4)  -sin(pi/4) ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.rmds = RotationMatrixDifferentiationSystem();
            
            % Actual values
            coefActual = step(testCase.rmds,pi/4,1,1);
            
            % Evaluation
            testCase.verifyEqual(coefExpctd,coefActual,'RelTol',1e-10);
        end
        
        % Test for default construction
        function testConstructorWithAnglesAndMus(testCase)
            
            % Expected values
            coefExpctd = [
                -sin(pi/4) -cos(pi/4) ;
                -cos(pi/4)  sin(pi/4) ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.rmds = RotationMatrixDifferentiationSystem();
            
            % Actual values
            coefActual = step(testCase.rmds,pi/4,[ 1 -1 ],1);            
            
            % Evaluation
            testCase.verifyEqual(coefExpctd,coefActual,'RelTol',1e-10);
            
        end

        % Test for set angle
        function testSetAngles(testCase)
            
            % Expected values
            coefExpctd = [
                0 -1 ;
                1  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.rmds = RotationMatrixDifferentiationSystem();
            
            % Actual values
            coefActual = step(testCase.rmds,0,1,1);
            
            % Evaluation
            testCase.verifyEqual(coefExpctd,coefActual,'AbsTol',1e-10);
            
            % Expected values
            coefExpctd = [
                -sin(pi/4) -cos(pi/4) ;
                 cos(pi/4) -sin(pi/4) ];
            
            % Actual values
            coefActual = step(testCase.rmds,pi/4,1,1);
            
            % Evaluation
            testCase.verifyEqual(coefExpctd,coefActual,'RelTol',1e-10);
        end
        
        % Test for set angle
        function test4x4(testCase)
            
            % Expected values*
            angs = zeros(6,1);
            angs(1) = randn();
            G1 = zeros(4); 
            G1(1,1) = -sin(angs(1));
            G1(1,2) = -cos(angs(1));
            G1(2,1) = cos(angs(1));
            G1(2,2) = -sin(angs(1));
            coefExpctd = G1;
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.rmds = RotationMatrixDifferentiationSystem();
            
            % Actual values
            coefActual = step(testCase.rmds,angs,1,1);
            
            % Evaluation
            testCase.verifyEqual(coefExpctd,coefActual,'AbsTol',1e-10);

        end        
        
        function test5x5(testCase)

            %
            M = 5;
            nAngs = M*(M-1)/2; 
            idx = randi(10,1);
            
            % Expected values
            angs = randn(nAngs,1);
            coefExpctd = 1;
            ig = 1;
            for itop = 1:M-1
                for ibtm = itop+1:M
                    if ig == idx
                        G = zeros(M);                        
                        G(itop,itop) = -sin(angs(ig));
                        G(itop,ibtm) = -cos(angs(ig)); 
                        G(ibtm,itop) =  cos(angs(ig)); 
                        G(ibtm,ibtm) = -sin(angs(ig));                        
                    else
                        G = eye(M);                        
                        G(itop,itop) =  cos(angs(ig));
                        G(itop,ibtm) = -sin(angs(ig)); 
                        G(ibtm,itop) =  sin(angs(ig)); 
                        G(ibtm,ibtm) =  cos(angs(ig));
                    end
                    ig = ig+1;
                    coefExpctd = G*coefExpctd;
                end
            end
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.rmds = RotationMatrixDifferentiationSystem();
            
            % Actual values
            coefActual = step(testCase.rmds,angs,1,idx);
            
            % Evaluation
            testCase.verifyEqual(coefExpctd,coefActual,'AbsTol',1e-10);

        end          
        
        function test8x8(testCase)

            %
            M = 8;
            nAngs = M*(M-1)/2; 
            idx = randi(10,1);
            
            % Expected values
            angs = randn(nAngs,1);
            coefExpctd = 1;
            ig = 1;
            for itop = 1:M-1
                for ibtm = itop+1:M
                    if ig == idx
                        G = zeros(M);                        
                        G(itop,itop) = -sin(angs(ig));
                        G(itop,ibtm) = -cos(angs(ig)); 
                        G(ibtm,itop) =  cos(angs(ig)); 
                        G(ibtm,ibtm) = -sin(angs(ig));                        
                    else
                        G = eye(M);                        
                        G(itop,itop) =  cos(angs(ig));
                        G(itop,ibtm) = -sin(angs(ig)); 
                        G(ibtm,itop) =  sin(angs(ig)); 
                        G(ibtm,ibtm) =  cos(angs(ig));
                    end
                    ig = ig+1;
                    coefExpctd = G*coefExpctd;
                end
            end
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.rmds = RotationMatrixDifferentiationSystem();
            
            % Actual values
            coefActual = step(testCase.rmds,angs,1,idx);
            
            % Evaluation
            testCase.verifyEqual(coefExpctd,coefActual,'AbsTol',1e-10);

        end                  

        function test10x10(testCase)

            %
            M = 10;
            nAngs = M*(M-1)/2; 
            idx = randi(10,1);
            
            % Expected values
            angs = randn(nAngs,1);
            coefExpctd = 1;
            ig = 1;
            for itop = 1:M-1
                for ibtm = itop+1:M
                    if ig == idx
                        G = zeros(M);                        
                        G(itop,itop) = -sin(angs(ig));
                        G(itop,ibtm) = -cos(angs(ig)); 
                        G(ibtm,itop) =  cos(angs(ig)); 
                        G(ibtm,ibtm) = -sin(angs(ig));                        
                    else
                        G = eye(M);                        
                        G(itop,itop) =  cos(angs(ig));
                        G(itop,ibtm) = -sin(angs(ig)); 
                        G(ibtm,itop) =  sin(angs(ig)); 
                        G(ibtm,ibtm) =  cos(angs(ig));
                    end
                    ig = ig+1;
                    coefExpctd = G*coefExpctd;
                end
            end
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.rmds = RotationMatrixDifferentiationSystem();
            
            % Actual values
            coefActual = step(testCase.rmds,angs,1,idx);
            
            % Evaluation
            testCase.verifyEqual(coefExpctd,coefActual,'AbsTol',1e-10);

        end                  
        
    end
end
