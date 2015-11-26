classdef DegradationSystemTestCase < matlab.unittest.TestCase
    %BLURSYSTEMTESTCASE Test case for BlurSystem
    %
    % SVN identifier:
    % $Id: DegradationSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627    
    %
    properties
        degradation
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.degradation);
        end
    end
    
    methods (Test)
        
        % Test for default construction
        function testGetLinearProcess(testCase)

            % Expected values
            import saivdr.degradation.linearprocess.*
            linProcExpctd = BlurSystem();
            procModeExpctd = 'Normal';
            
            % Instantiation of target class
            import saivdr.degradation.*
            testCase.degradation = DegradationSystem(...
                'LinearProcess',linProcExpctd);
            
            % Actual values
            linProcActual = get(testCase.degradation,'LinearProcess');                      
            procModeActual = get(linProcActual,'ProcessingMode');
            
            % Evaluation
            testCase.assertEqual(linProcActual,linProcExpctd);
            testCase.assertEqual(procModeActual,procModeExpctd);
            
        end 

    end

end
