classdef ModulePartialLineBufferTestCase < matlab.unittest.TestCase
    %MODULEPARTIALLINEBUFFERTESTCASE Test case for ModulePartialLineBuffer
    %
    % SVN identifier:
    % $Id: ModulePartialLineBufferTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        
        function testPartialLineBuffer2plus2withBottom2ChDelay(testCase)

            nBlks = 4;
            csIn = [ 1 2 ; 3 4 ; 5 6 ; 7 8 ].';
            caIn = [ 9 0 ; 1 2 ; 3 4 ; 5 6 ].';
            lineLength = 2;
            
            % Expected values
            csOutExpctd = [ 1 2 ; 3 4 ; 5 6 ; 7 8 ].';
            caOutExpctd = [ 0 0 ; 0 0 ; 9 0 ; 1 2 ].';
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialLineBuffer(...
                'LineLength',lineLength);
            
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

        function testPartialLineBuffer2plus2withBottom2ChDelayRandom(testCase)

            nBlks = 8;
            csIn = rand(2,nBlks);
            caIn = rand(2,nBlks);
            lineLength = 4;
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = [zeros(2,4) caIn(:,1:nBlks-4)];
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialLineBuffer(...
                'LineLength', lineLength);
            
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

        function testPartialLineBuffer2plus2withTop2ChDelay(testCase)

            nBlks = 4;
            csIn = [ 1 2 ; 3 4 ; 5 6 ; 7 8 ].';
            caIn = [ 9 0 ; 1 2 ; 3 4 ; 5 6 ].';
            startIndexOfDelayChannel = 1;
            numberOfDelayChannels = 2;
            lineLength = 2;
            
            % Expected values
            csOutExpctd = [ 0 0 ; 0 0 ; 1 2 ; 3 4 ].';
            caOutExpctd = [ 9 0 ; 1 2 ; 3 4 ; 5 6 ].';
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialLineBuffer(...
                'LineLength', lineLength,...
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

        function testPartialLineBuffer2plus2withTop2ChDelayRandom(testCase)

            nBlks = 8;
            csIn = rand(2,nBlks);
            caIn = rand(2,nBlks);
            startIndexOfDelayChannel = 1;
            numberOfDelayChannels = 2;
            lineLength = 4;
            
            % Expected values
            csOutExpctd = [zeros(2,4) csIn(:,1:nBlks-4)];
            caOutExpctd = caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialLineBuffer(...
                'LineLength',lineLength,...
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

        function testPartialLineBuffer5plus2withBottom2ChDelayRandom(testCase)

            nBlks = 8;
            csIn = rand(5,nBlks);
            caIn = rand(2,nBlks);
            startIndexOfDelayChannel = 6;
            numberOfDelayChannels = 2;
            lineLength = 4;
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = [zeros(2,4) caIn(:,1:nBlks-4)];
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialLineBuffer(...
                'LineLength',lineLength,...
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

        function testPartialLineBuffer5plus2withTop2ChDelayRandom(testCase)

            nBlks = 16;
            csIn = rand(5,nBlks);
            caIn = rand(2,nBlks);
            startIndexOfDelayChannel = 1;
            numberOfDelayChannels = 2;
            lineLength = 4;
            
            % Expected values
            csOutExpctd(1:2,:) = [zeros(2,4) csIn(1:2,1:nBlks-4)];
            csOutExpctd(3:5,:) = csIn(3:5,:);
            caOutExpctd = caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialLineBuffer(...
                'LineLength',lineLength,...
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

            nBlks = 32;
            csIn = rand(5,nBlks);
            caIn = rand(2,nBlks);
            startIndexOfDelayChannel = 3;
            numberOfDelayChannels = 5;
            lineLength = 8;
            
            % Expected values
            csOutExpctd(1:2,:) = csIn(1:2,:);
            csOutExpctd(3:5,:) = [zeros(3,8) csIn(3:5,1:nBlks-8)];
            caOutExpctd = [zeros(2,8) caIn(:,1:nBlks-8)];
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialLineBuffer(...
                'LineLength',lineLength,...
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

            nBlks = 32;
            csIn = rand(5,nBlks);
            caIn = rand(2,nBlks);
            startIndexOfDelayChannel = 1;
            numberOfDelayChannels = 5;
            lineLength = 4;
            
            % Expected values
            csOutExpctd = [zeros(5,4) csIn(:,1:nBlks-4)];
            caOutExpctd = caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModulePartialLineBuffer(...
                'LineLength',lineLength,...
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
          
