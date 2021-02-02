classdef ModulePartialDelayTestCase < matlab.unittest.TestCase
    %MODULEHORIZONTALDELAYTESTCASE Test case for ModulePartialDelay
    %
    % SVN identifier:
    % $Id: ModulePartialDelayTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        
        function testPartialDelay2plus2withBottom2ChDelay(testCase)

            csIn = [ 1 2 ; 3 4 ].';
            caIn = [ 5 6 ; 7 8 ].';
            
            % Expected values
            csOutExpctd = [ 1 2 ; 3 4 ].';
            caOutExpctd = [ 0 0 ; 5 6 ].';
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialDelay();
            
            % Actual values
            [csOutActual(:,1),caOutActual(:,1)] = step(testCase.module,csIn(:,1),caIn(:,1));
            [csOutActual(:,2),caOutActual(:,2)] = step(testCase.module,csIn(:,2),caIn(:,2));
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));            
         end
        
        function testPartialDelay2plus2withBottom2ChDelayRandom(testCase)

            nBlks = 4;
            csIn = rand(2,nBlks);
            caIn = rand(2,nBlks);
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = [zeros(2,1) caIn(:,1:nBlks-1)];
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialDelay();
            
            % Actual 
            csOutActual = zeros(2,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx));
            end
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));            
        end
        
        function testPartialDelay2plus2withTop2ChDelay(testCase)

            csIn = [ 1 2 ; 3 4 ].';
            caIn = [ 5 6 ; 7 8 ].';
            startIndexOfDelayChannel = 1;
            numberOfDelayChannels = 2;
            
            % Expected values
            csOutExpctd = [ 0 0 ; 1 2 ].';
            caOutExpctd = [ 5 6 ; 7 8 ].';
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialDelay(...
                'StartIndexOfDelayChannel',startIndexOfDelayChannel,...
                'NumberOfDelayChannels',numberOfDelayChannels);
            
            % Actual values
            [csOutActual(:,1),caOutActual(:,1)] = step(testCase.module,csIn(:,1),caIn(:,1));
            [csOutActual(:,2),caOutActual(:,2)] = step(testCase.module,csIn(:,2),caIn(:,2));
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));     
            
        end
         
        function testPartialDelay2plus2withTop2ChDelayRandom(testCase)

            nBlks = 4;
            csIn = rand(2,nBlks);
            caIn = rand(2,nBlks);
            startIndexOfDelayChannel = 1;
            numberOfDelayChannels = 2;
            
            % Expected values
            csOutExpctd = [zeros(2,1) csIn(:,1:nBlks-1)];
            caOutExpctd = caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialDelay(...
                'StartIndexOfDelayChannel',startIndexOfDelayChannel,...
                'NumberOfDelayChannels',numberOfDelayChannels);
            
            % Actual values
            csOutActual = zeros(2,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx=1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx));
            end
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));     
            
        end
        
        function testPartialDelay5plus2withBottom2ChDelayRandom(testCase)

            nBlks = 4;
            csIn = rand(5,nBlks);
            caIn = rand(2,nBlks);
            startIndexOfDelayChannel = 6;
            numberOfDelayChannels = 2;
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = [zeros(2,1) caIn(:,1:nBlks-1)];
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialDelay(...
                'NumberOfSymmetricChannels',5,...
                'NumberOfAntisymmetricChannels',2,...
                'StartIndexOfDelayChannel',startIndexOfDelayChannel,...
                'NumberOfDelayChannels',numberOfDelayChannels);
            
            % Actual 
            csOutActual = zeros(5,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx));
            end
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));            
        end

        function testPartialDelay5plus2withTop2ChDelayRandom(testCase)

            nBlks = 4;
            csIn = rand(5,nBlks);
            caIn = rand(2,nBlks);
            startIndexOfDelayChannel = 1;
            numberOfDelayChannels = 2;
            
            % Expected values
            csOutExpctd(1:2,:) = [zeros(2,1) csIn(1:2,1:nBlks-1)];
            csOutExpctd(3:5,:) = csIn(3:5,:);
            caOutExpctd = caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialDelay(...
                'NumberOfSymmetricChannels',5,...
                'NumberOfAntisymmetricChannels',2,...                
                'StartIndexOfDelayChannel',startIndexOfDelayChannel,...
                'NumberOfDelayChannels',numberOfDelayChannels);
            
            % Actual values
            csOutActual = zeros(5,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx=1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx));
            end
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));     
            
        end

        function testPartialDelay5plus2withBottom5ChDelayRandom(testCase)

            nBlks = 4;
            csIn = rand(5,nBlks);
            caIn = rand(2,nBlks);
            startIndexOfDelayChannel = 3;
            numberOfDelayChannels = 5;
            
            % Expected values
            csOutExpctd(1:2,:) = csIn(1:2,:);
            csOutExpctd(3:5,:) = [zeros(3,1) csIn(3:5,1:nBlks-1)];
            caOutExpctd = [zeros(2,1) caIn(:,1:nBlks-1)];
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialDelay(...
                'NumberOfSymmetricChannels',5,...
                'NumberOfAntisymmetricChannels',2,...                
                'StartIndexOfDelayChannel',startIndexOfDelayChannel,...
                'NumberOfDelayChannels',numberOfDelayChannels);
            
            % Actual 
            csOutActual = zeros(5,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx));
            end
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));            
        end

        function testPartialDelay5plus2withTop5ChDelayRandom(testCase)

            nBlks = 4;
            csIn = rand(5,nBlks);
            caIn = rand(2,nBlks);
            startIndexOfDelayChannel = 1;
            numberOfDelayChannels = 5;
            
            % Expected values
            csOutExpctd = [zeros(5,1) csIn(:,1:nBlks-1)];
            caOutExpctd = caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialDelay(...
                'NumberOfSymmetricChannels',5,...
                'NumberOfAntisymmetricChannels',2,...                
                'StartIndexOfDelayChannel',startIndexOfDelayChannel,...
                'NumberOfDelayChannels',numberOfDelayChannels);
            
            % Actual values
            csOutActual = zeros(5,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx=1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx));
            end
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));     
            
        end
        
    end
end
          
