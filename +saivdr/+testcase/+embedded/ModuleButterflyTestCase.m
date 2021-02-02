classdef ModuleButterflyTestCase < matlab.unittest.TestCase
    %MODULEBUTTERFLYTESTCASE Test case for ModuleButterflyDelay
    %
    % SVN identifier:
    % $Id: ModuleButterflyTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        module
    end
 
    methods(TestMethodSetup)
      % function createFigure(testCase)
      %     testCase.TestFigure = figure;
      % end
    end
 
    methods(TestMethodTeardown)
      function deleteObject(testCase)
          delete(testCase.module);
      end
    end
 
    methods(Test)
        
        function testButterfly2plus2(testCase)

            csIn = [ 1 2 ].';
            caIn = [ 3 4 ].';
            
            % Expected values
            csOutExpctd = [  2.828427124746190 4.242640687119285 ].';
            caOutExpctd = [ -1.414213562373095 -1.414213562373095 ].';
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleButterfly();
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn);
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));    
            
         end

        function testButterfly2plus2Random(testCase)

            nChs = [ 2 2 ];
            csIn = rand(nChs(1),1);
            caIn = rand(nChs(2),1);
            
            % Expected values
            csOutExpctd = (csIn + caIn)/sqrt(2);
            caOutExpctd = (csIn - caIn)/sqrt(2);            

            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleButterfly();
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn);
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));            
        end
        
        function testButterfly2plus2SuccessiveProcessing(testCase)

            csIn = [ 1 2 ; 3 4 ; 5 6 ; 7 8 ].'; 
            caIn = [ 9 0 ; 1 2 ; 3 4 ; 5 6 ].';
            nBlks = size(csIn,2);
            
            % Expected values
            csOutExpctd = [ 10 2 ; 4 6 ; 8 10 ; 12 14 ].'/sqrt(2);
            caOutExpctd = [ -8 2 ; 2 2 ; 2  2 ;  2  2 ].'/sqrt(2); 
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleButterfly();
            
            % Actual values
            csOutActual = zeros(2,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx));
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csOutExpctd(:,idx) - csOutActual(:,idx))./abs(csOutExpctd(:,idx)));
                testCase.verifyEqual(csOutActual(:,idx),csOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caOutExpctd(:,idx) - caOutActual(:,idx))./abs(caOutExpctd(:,idx)));
                testCase.verifyEqual(caOutActual(:,idx),caOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
            
        end

        function testButterfly2plus2SuccessiveProcessingRandom(testCase)

            nBlks = 4;
            csIn = rand(2,nBlks);
            caIn = rand(2,nBlks);
            
            % Expected values
            csOutExpctd = (csIn+caIn)/sqrt(2);
            caOutExpctd = (csIn-caIn)/sqrt(2);
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleButterfly();
            
            % Actual values
            csOutActual = zeros(2,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx));
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csOutExpctd(:,idx) - csOutActual(:,idx))./abs(csOutExpctd(:,idx)));
                testCase.verifyEqual(csOutActual(:,idx),csOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caOutExpctd(:,idx) - caOutActual(:,idx))./abs(caOutExpctd(:,idx)));
                testCase.verifyEqual(caOutActual(:,idx),caOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end            
        end        

        function testButterfly5plus2(testCase)
            
            csIn = [ 1 2 3 4 5 ].';
            caIn = [ 6 7 ].';
            
            % Expected values
            csOutExpctd = [  4.949747468305833   6.363961030678928 3 4 5 ].';
            caOutExpctd = [ -3.535533905932737  -3.535533905932737 ].';
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleButterfly();
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn);
            
            % Evaluation
            testCase.verifySize(csOutActual,size(csOutExpctd));
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifySize(caOutActual,size(caOutExpctd));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));      
        
        end

        function testButterfly5plus2Random(testCase)

            csIn = rand(5,1);
            caIn = rand(2,1);
            
            % Expected values
            csOutExpctd = csIn;
            csOutExpctd(1:2) = (csIn(1:2)+caIn)/sqrt(2);
            caOutExpctd = (csIn(1:2)-caIn)/sqrt(2);
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleButterfly();
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn);
            
            % Evaluation
            testCase.verifySize(csOutActual,size(csOutExpctd));
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifySize(caOutActual,size(caOutExpctd));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end

    end
 
end
