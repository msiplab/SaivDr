classdef OrthonormalMatrixFactorizationSystemTestCase < ...
        matlab.unittest.TestCase
    %ORTHONORMALMATRIXFACTORIZATIONSYSTEMTESTCASE Test case for OrthonormalMatrixFactorizationSystem
    %
    % SVN identifier:
    % $Id: OrthonormalMatrixFactorizationSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        omfs
    end
    
    methods (TestMethodTeardown) 
        function deleteObject(testCase)
            delete(testCase.omfs);
        end
    end
    
    methods (Test)

        % Test for set angle
        function testMatrixFactorization2x2(testCase)
            
            % Expected values
            mtxExpctd = [
                0 -1 ;
                1  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.omfs = OrthonormalMatrixFactorizationSystem();
            omgs = OrthonormalMatrixGenerationSystem();
            
            % Get matrix
            [angles,mus] = step(testCase.omfs,mtxExpctd);
            mtxActual = step(omgs,angles,mus);
            
            % Evaluation
            message = 'Actual matrix is not the expected one.';
            testCase.verifyEqual(mtxActual,mtxExpctd,'AbsTol',1e-15,message);
        
        end
  
        % Test for set angle
        function testMatrixFactorization4x4(testCase)
            
            % Expected values
            mtxExpctd = [
                0  0  1  0 ;
                0  0  0  1 ;
                1  0  0  0 ;
                0  1  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*       
            testCase.omfs = OrthonormalMatrixFactorizationSystem();
            omgs = OrthonormalMatrixGenerationSystem();
            
            % Get matrix
            [angles,mus] = step(testCase.omfs,mtxExpctd);
            mtxActual = step(omgs,angles,mus);
            
            % Evaluation
            message = 'Actual matrix is not the expected one.';
            testCase.verifyEqual(mtxActual,mtxExpctd,'AbsTol',1e-15,message);
        end
        
        % Test for set angle
        function testMatrixFactorization8x8(testCase)
            
            % Expected values
            I = eye(4);
            Z = zeros(4);
            mtxExpctd = [
                Z  I ;
                I  Z ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.omfs = OrthonormalMatrixFactorizationSystem();
            omgs = OrthonormalMatrixGenerationSystem();
            
            % Get matrix
            [angles,mus] = step(testCase.omfs,mtxExpctd);
            mtxActual = step(omgs,angles,mus);
            
            % Evaluation
            message = 'Actual matrix is not the expected one.';
            testCase.verifyEqual(mtxActual,mtxExpctd,'AbsTol',1e-15,message);
        end
                
    end
end
