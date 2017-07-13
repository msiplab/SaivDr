classdef OrthogonalProjectionSystemTestCase < matlab.unittest.TestCase
    %ORTHOGONALPROJECTIONSYSTEMTESTCASE Test case for OrthogonalProjectionSystem
    %
    % SVN identifier:
    % $Id: OrthogonalProjectionSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        ops
    end
    
    methods (TestMethodTeardown) 
        function deleteObject(testCase)
            delete(testCase.ops);
        end
    end
    
    methods (Test)
        
        % Test for set angle
        function testProjectionOfVector(testCase)
            
            angleExpctd = -pi/4;
            
            % Vector
            vector = [ 1 -1 ; 1  1 ] * [ 1 ; 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            testCase.ops = OrthogonalProjectionSystem();
            
            % Set vector
            angleActual = step(testCase.ops,vector);
                        
            % Evaluation
            message = 'Actual angle is not the expected one.';
            testCase.verifyEqual(angleActual,angleExpctd,'RelTol',1e-15,message);
        end
        
    end
end
