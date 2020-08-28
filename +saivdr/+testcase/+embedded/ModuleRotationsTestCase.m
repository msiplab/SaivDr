classdef ModuleRotationsTestCase < matlab.unittest.TestCase
    %MODULEROTATIONSTESTCASE Test case for ModuleRotation
    %
    % SVN identifier:
    % $Id: ModuleRotationsTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        
        function testRotations2plus2WU(testCase)
            
            csIn = [ 1 2 ].';
            caIn = [ 3 4 ].';
            matrixW = [ 1 -1 ;  1 1 ]/sqrt(2);
            matrixU = [ 1  1 ; -1 1 ]/sqrt(2);
            isTermination = false;
            
            % Expected values
            csOutExpctd = [ -0.707106781186547 2.121320343559643 ].';
            caOutExpctd = [  4.949747468305833 0.707106781186548 ].';
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixW',matrixW,...
                'MatrixU',matrixU);
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn,isTermination);
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testRotations2plus2WURandom(testCase)
            
            nChs = [ 2 2 ];
            csIn = rand(nChs(1),1);
            caIn = rand(nChs(2),1);
            rotation = @(theta) [ cos(theta) sin(theta) ; -sin(theta) cos(theta) ];
            matrixW = rotation(rand());
            matrixU = rotation(rand());
            isTermination = false;
            
            % Expected values
            csOutExpctd = matrixW*csIn;
            caOutExpctd = matrixU*caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixW',matrixW,...
                'MatrixU',matrixU);
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn,isTermination);
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        function testRotations2plus2WUSuccessiveProcessing(testCase)
            
            csIn = [ 1 2 ; 3 4 ; 5 6 ; 7 8 ].';
            caIn = [ 9 0 ; 1 2 ; 3 4 ; 5 6 ].';
            nBlks = size(csIn,2);
            matrixW = [ 1 -1 ;  1 1 ]/sqrt(2);
            matrixU = [ 1  1 ; -1 1 ]/sqrt(2);
            isTermination = false;
            
            % Expected values
            csOutExpctd = [ -1 3 ; -1 7 ; -1 11 ; -1 15 ].'/sqrt(2);
            caOutExpctd = [ 9 -9 ;  3 1 ;  7  1 ; 11  1 ].'/sqrt(2);
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixW',matrixW,...
                'MatrixU',matrixU);
            
            % Actual values
            csOutActual = zeros(2,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csOutExpctd(:,idx) - csOutActual(:,idx))./abs(csOutExpctd(:,idx)));
                testCase.verifyEqual(csOutActual(:,idx),csOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caOutExpctd(:,idx) - caOutActual(:,idx))./abs(caOutExpctd(:,idx)));
                testCase.verifyEqual(caOutActual(:,idx),caOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
            
        end
        
        function testRotations2plus2WUSuccessiveProcessingRandom(testCase)
            
            nBlks = 4;
            csIn = rand(2,nBlks);
            caIn = rand(2,nBlks);
            rotation = @(theta) [ cos(theta) sin(theta) ; -sin(theta) cos(theta) ];
            matrixW = rotation(rand());
            matrixU = rotation(rand());
            isTermination = false;
            
            % Expected values
            csOutExpctd = matrixW * csIn;
            caOutExpctd = matrixU * caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixW', matrixW,...
                'MatrixU', matrixU);
            
            % Actual values
            csOutActual = zeros(2,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csOutExpctd(:,idx) - csOutActual(:,idx))./abs(csOutExpctd(:,idx)));
                testCase.verifyEqual(csOutActual(:,idx),csOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caOutExpctd(:,idx) - caOutActual(:,idx))./abs(caOutExpctd(:,idx)));
                testCase.verifyEqual(caOutActual(:,idx),caOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
        end
        
        function testRotations2plus2IU(testCase)
            
            csIn = [ 1 2 ].';
            caIn = [ 3 4 ].';
            matrixU = [ 1  1 ; -1 1 ]/sqrt(2);
            isTermination = false;
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = [  4.949747468305833 0.707106781186548 ].';
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixU',matrixU);
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn,isTermination);
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testRotations2plus2IURandom(testCase)
            
            nChs = [ 2 2 ];
            csIn = rand(nChs(1),1);
            caIn = rand(nChs(2),1);
            rotation = @(theta) [ cos(theta) sin(theta) ; -sin(theta) cos(theta) ];
            matrixU = rotation(rand());
            isTermination = false;
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = matrixU*caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixU',matrixU);
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn,isTermination);
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        function testRotations2plus2IUSuccessiveProcessing(testCase)
            
            csIn = [ 1 2 ; 3 4 ; 5 6 ; 7 8 ].';
            caIn = [ 9 0 ; 1 2 ; 3 4 ; 5 6 ].';
            nBlks = size(csIn,2);
            matrixU = [ 1  1 ; -1 1 ]/sqrt(2);
            isTermination = false;
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = [ 9 -9 ;  3 1 ;  7  1 ; 11  1 ].'/sqrt(2);
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixU',matrixU);
            
            % Actual values
            csOutActual = zeros(2,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csOutExpctd(:,idx) - csOutActual(:,idx))./abs(csOutExpctd(:,idx)));
                testCase.verifyEqual(csOutActual(:,idx),csOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caOutExpctd(:,idx) - caOutActual(:,idx))./abs(caOutExpctd(:,idx)));
                testCase.verifyEqual(caOutActual(:,idx),caOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
            
        end
        
        function testRotations2plus2IUSuccessiveProcessingRandom(testCase)
            
            nBlks = 4;
            csIn = rand(2,nBlks);
            caIn = rand(2,nBlks);
            rotation = @(theta) [ cos(theta) sin(theta) ; -sin(theta) cos(theta) ];
            matrixU = rotation(rand());
            isTermination = false;
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = matrixU * caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixU', matrixU);
            
            % Actual values
            csOutActual = zeros(2,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csOutExpctd(:,idx) - csOutActual(:,idx))./abs(csOutExpctd(:,idx)));
                testCase.verifyEqual(csOutActual(:,idx),csOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caOutExpctd(:,idx) - caOutActual(:,idx))./abs(caOutExpctd(:,idx)));
                testCase.verifyEqual(caOutActual(:,idx),caOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
        end
        
        function testRotations2plus2IUWithTermination(testCase)
            
            csIn = [ 1 2 ].';
            caIn = [ 3 4 ].';
            matrixU = [ 1  1 ; -1 1 ]/sqrt(2);
            isTermination = true;
            
            % Expected values
            csOutExpctd =  csIn;
            caOutExpctd = -caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixU',matrixU);
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn,isTermination);
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testRotations2plus2IUSuccessiveProcessingWithTermination(testCase)
            
            csIn = [ 1 2 ; 3 4 ; 5 6 ; 7 8 ].';
            caIn = [ 9 0 ; 1 2 ; 3 4 ; 5 6 ].';
            nBlks = size(csIn,2);
            matrixU = [ 1  1 ; -1 1 ]/sqrt(2);
            
            % Expected values (Termination {on, off, on, off}
            csOutExpctd = csIn;
            caOutExpctd = [ -9 0 ; 2.121320343559642  0.707106781186547; -3 -4 ;  7.778174593052023  0.707106781186547 ].';
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixU',matrixU);
            
            % Actual values
            csOutActual = zeros(2,nBlks);
            caOutActual = zeros(2,nBlks);
            
            idx = 1;
            isTermination = true;
            [csOutActual(:,idx),caOutActual(:,idx)] = ...
                step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
            
            idx = 2;
            isTermination = false;
            [csOutActual(:,idx),caOutActual(:,idx)] = ...
                step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
            
            idx = 3;
            isTermination = true;
            [csOutActual(:,idx),caOutActual(:,idx)] = ...
                step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
            
            idx = 4;
            isTermination = false;
            [csOutActual(:,idx),caOutActual(:,idx)] = ...
                step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csOutExpctd(:,idx) - csOutActual(:,idx))./abs(csOutExpctd(:,idx)));
                testCase.verifyEqual(csOutActual(:,idx),csOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caOutExpctd(:,idx) - caOutActual(:,idx))./abs(caOutExpctd(:,idx)));
                testCase.verifyEqual(caOutActual(:,idx),caOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
            
        end
        
        function testRotations2plus2IUSuccessiveProcessingRandomWithTermination(testCase)
            
            nBlks = 4;
            csIn = rand(2,nBlks);
            caIn = rand(2,nBlks);
            rotation = @(theta) [ cos(theta) sin(theta) ; -sin(theta) cos(theta) ];
            matrixU = rotation(rand());
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = zeros(2,nBlks);
            isTermination = true;
            for idx = 1:nBlks
                if isTermination
                    caOutExpctd(:,idx) = -caIn(:,idx);
                else
                    caOutExpctd(:,idx) = matrixU*caIn(:,idx);
                end
                isTermination = ~isTermination;
            end
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixU', matrixU);
            
            % Actual values
            csOutActual = zeros(2,nBlks);
            caOutActual = zeros(2,nBlks);
            isTermination = true;
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
                isTermination = ~isTermination;
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csOutExpctd(:,idx) - csOutActual(:,idx))./abs(csOutExpctd(:,idx)));
                testCase.verifyEqual(csOutActual(:,idx),csOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caOutExpctd(:,idx) - caOutActual(:,idx))./abs(caOutExpctd(:,idx)));
                testCase.verifyEqual(caOutActual(:,idx),caOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
        end
        
        function testRotations5plus2WU(testCase)
            
            csIn = [ 1 2 3 4 5 ].';
            caIn = [ 6 7 ].';
            matrixW = dctmtx(5);
            matrixU = [ 1  1 ; -1 1 ]/sqrt(2);
            isTermination = false;
            
            % Expected values
            csOutExpctd = [
                6.708203932499369
                -3.149499888950551
                0
                -0.283990227825647
                0 ];
            caOutExpctd = [
                9.192388155425117
                0.707106781186547 ];
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixW', matrixW,...
                'MatrixU', matrixU);
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn,isTermination);
            
            % Evaluation
            testCase.verifySize(csOutActual,size(csOutExpctd));
            diff = max(abs(csOutExpctd - csOutActual));
            testCase.verifyEqual(csOutActual,csOutExpctd,'AbsTol',1e-14,sprintf('%g',diff));
            testCase.verifySize(caOutActual,size(caOutExpctd));
            diff = max(abs(caOutExpctd - caOutActual));
            testCase.verifyEqual(caOutActual,caOutExpctd,'AbsTol',1e-14,sprintf('%g',diff));
            
        end
        
        function testRotations5plus2WURandom(testCase)
            
            csIn = rand(5,1);
            caIn = rand(2,1);
            rotation = @(theta) [ cos(theta) sin(theta) ; -sin(theta) cos(theta) ];
            matrixW = eye(5);
            matrixW(1:2,:) = rotation(rand())*matrixW(1:2,:);
            matrixW(2:3,:) = rotation(rand())*matrixW(2:3,:);
            matrixW(3:4,:) = rotation(rand())*matrixW(3:4,:);
            matrixW(4:5,:) = rotation(rand())*matrixW(4:5,:);
            matrixU = rotation(rand());
            isTermination = false;
            
            % Expected values
            csOutExpctd = matrixW*csIn;
            caOutExpctd = matrixU*caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixW', matrixW,...
                'MatrixU', matrixU);
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn,isTermination);
            
            % Evaluation
            testCase.verifySize(csOutActual,size(csOutExpctd));
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifySize(caOutActual,size(caOutExpctd));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testRotations5plus2WUSuccessiveProcessingRandom(testCase)
            
            nBlks = 4;
            csIn = rand(5,nBlks);
            caIn = rand(2,nBlks);
            rotation = @(theta) [ cos(theta) sin(theta) ; -sin(theta) cos(theta) ];
            matrixW = eye(5);
            matrixW(1:2,:) = rotation(rand())*matrixW(1:2,:);
            matrixW(2:3,:) = rotation(rand())*matrixW(2:3,:);
            matrixW(3:4,:) = rotation(rand())*matrixW(3:4,:);
            matrixW(4:5,:) = rotation(rand())*matrixW(4:5,:);
            matrixU = rotation(rand());
            isTermination = false;
            
            % Expected values
            csOutExpctd = matrixW * csIn;
            caOutExpctd = matrixU * caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixW', matrixW,...
                'MatrixU', matrixU);
            
            % Actual values
            csOutActual = zeros(5,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csOutExpctd(:,idx) - csOutActual(:,idx))./abs(csOutExpctd(:,idx)));
                testCase.verifyEqual(csOutActual(:,idx),csOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caOutExpctd(:,idx) - caOutActual(:,idx))./abs(caOutExpctd(:,idx)));
                testCase.verifyEqual(caOutActual(:,idx),caOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
        end
        
        function testRotations5plus2IU(testCase)
            
            csIn = [ 1 2 3 4 5 ].';
            caIn = [ 6 7 ].';
            matrixU = [ 1  1 ; -1 1 ]/sqrt(2);
            isTermination = false;
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = [
                9.192388155425117
                0.707106781186547 ];
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixU', matrixU);
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn,isTermination);
            
            % Evaluation
            testCase.verifySize(csOutActual,size(csOutExpctd));
            diff = max(abs(csOutExpctd - csOutActual));
            testCase.verifyEqual(csOutActual,csOutExpctd,'AbsTol',1e-14,sprintf('%g',diff));
            testCase.verifySize(caOutActual,size(caOutExpctd));
            diff = max(abs(caOutExpctd - caOutActual));
            testCase.verifyEqual(caOutActual,caOutExpctd,'AbsTol',1e-14,sprintf('%g',diff));
            
        end
        
        function testRotations5plus2IURandom(testCase)
            
            csIn = rand(5,1);
            caIn = rand(2,1);
            rotation = @(theta) [ cos(theta) sin(theta) ; -sin(theta) cos(theta) ];
            matrixU = rotation(rand());
            isTermination = false;
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = matrixU*caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixU', matrixU);
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn,isTermination);
            
            % Evaluation
            testCase.verifySize(csOutActual,size(csOutExpctd));
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifySize(caOutActual,size(caOutExpctd));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testRotations5plus2IUSuccessiveProcessingRandom(testCase)
            
            nBlks = 4;
            csIn = rand(5,nBlks);
            caIn = rand(2,nBlks);
            rotation = @(theta) [ cos(theta) sin(theta) ; -sin(theta) cos(theta) ];
            matrixU = rotation(rand());
            isTermination = false;
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = matrixU * caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixU', matrixU);
            
            % Actual values
            csOutActual = zeros(5,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csOutExpctd(:,idx) - csOutActual(:,idx))./abs(csOutExpctd(:,idx)));
                testCase.verifyEqual(csOutActual(:,idx),csOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caOutExpctd(:,idx) - caOutActual(:,idx))./abs(caOutExpctd(:,idx)));
                testCase.verifyEqual(caOutActual(:,idx),caOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
        end
        
        function testRotations5plus2WI(testCase)
            
            csIn = [ 1 2 3 4 5 ].';
            caIn = [ 6 7 ].';
            matrixW = dctmtx(5);
            isTermination = false;
            
            % Expected values
            csOutExpctd = [
                6.708203932499369
                -3.149499888950551
                0
                -0.283990227825647
                0 ];
            caOutExpctd = caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixW', matrixW);
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn,isTermination);
            
            % Evaluation
            testCase.verifySize(csOutActual,size(csOutExpctd));
            diff = max(abs(csOutExpctd - csOutActual));
            testCase.verifyEqual(csOutActual,csOutExpctd,'AbsTol',1e-14,sprintf('%g',diff));
            testCase.verifySize(caOutActual,size(caOutExpctd));
            diff = max(abs(caOutExpctd - caOutActual));
            testCase.verifyEqual(caOutActual,caOutExpctd,'AbsTol',1e-14,sprintf('%g',diff));
            
        end
        
        function testRotations5plus2WIRandom(testCase)
            
            csIn = rand(5,1);
            caIn = rand(2,1);
            rotation = @(theta) [ cos(theta) sin(theta) ; -sin(theta) cos(theta) ];
            matrixW = eye(5);
            matrixW(1:2,:) = rotation(rand())*matrixW(1:2,:);
            matrixW(2:3,:) = rotation(rand())*matrixW(2:3,:);
            matrixW(3:4,:) = rotation(rand())*matrixW(3:4,:);
            matrixW(4:5,:) = rotation(rand())*matrixW(4:5,:);
            isTermination = false;
            
            % Expected values
            csOutExpctd = matrixW*csIn;
            caOutExpctd = caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixW', matrixW);
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn,isTermination);
            
            % Evaluation
            testCase.verifySize(csOutActual,size(csOutExpctd));
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifySize(caOutActual,size(caOutExpctd));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testRotations5plus2WISuccessiveProcessingRandom(testCase)
            
            nBlks = 4;
            csIn = rand(5,nBlks);
            caIn = rand(2,nBlks);
            rotation = @(theta) [ cos(theta) sin(theta) ; -sin(theta) cos(theta) ];
            matrixW = eye(5);
            matrixW(1:2,:) = rotation(rand())*matrixW(1:2,:);
            matrixW(2:3,:) = rotation(rand())*matrixW(2:3,:);
            matrixW(3:4,:) = rotation(rand())*matrixW(3:4,:);
            matrixW(4:5,:) = rotation(rand())*matrixW(4:5,:);
            isTermination = false;
            
            % Expected values
            csOutExpctd = matrixW * csIn;
            caOutExpctd = caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixW', matrixW);
            
            % Actual values
            csOutActual = zeros(5,nBlks);
            caOutActual = zeros(2,nBlks);
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csOutExpctd(:,idx) - csOutActual(:,idx))./abs(csOutExpctd(:,idx)));
                testCase.verifyEqual(csOutActual(:,idx),csOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caOutExpctd(:,idx) - caOutActual(:,idx))./abs(caOutExpctd(:,idx)));
                testCase.verifyEqual(caOutActual(:,idx),caOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
        end
        
        function testRotations5plus2IUWithTermination(testCase)
            
            csIn = [ 1 2 3 4 5 ].';
            caIn = [ 6 7 ].';
            matrixU = [ 1  1 ; -1 1 ]/sqrt(2);
            isTermination = true;
            
            % Expected values
            csOutExpctd =  csIn;
            caOutExpctd = -caIn;
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixU',matrixU);
            
            % Actual values
            [csOutActual,caOutActual] = step(testCase.module,csIn,caIn,isTermination);
            
            % Evaluation
            diff = max(abs(csOutExpctd - csOutActual)./abs(csOutExpctd));
            testCase.verifyEqual(csOutActual,csOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caOutExpctd - caOutActual)./abs(caOutExpctd));
            testCase.verifyEqual(caOutActual,caOutExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
        function testRotations5plus2IUSuccessiveProcessingRandomWithTermination(testCase)
            
            nBlks = 4;
            csIn = rand(5,nBlks);
            caIn = rand(2,nBlks);
            rotation = @(theta) [ cos(theta) sin(theta) ; -sin(theta) cos(theta) ];
            matrixU = rotation(rand());
            
            % Expected values
            csOutExpctd = csIn;
            caOutExpctd = zeros(2,nBlks);
            isTermination = true;
            for idx=1:nBlks
                if isTermination
                    caOutExpctd(:,idx) = -caIn(:,idx);
                else
                    caOutExpctd(:,idx) = matrixU * caIn(:,idx);
                end
                isTermination = ~isTermination;
            end
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleRotations(...
                'MatrixU', matrixU);
            
            % Actual values
            csOutActual = zeros(5,nBlks);
            caOutActual = zeros(2,nBlks);
            isTermination = true;
            for idx = 1:nBlks
                [csOutActual(:,idx),caOutActual(:,idx)] = ...
                    step(testCase.module,csIn(:,idx),caIn(:,idx),isTermination);
                isTermination = ~isTermination;
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csOutExpctd(:,idx) - csOutActual(:,idx))./abs(csOutExpctd(:,idx)));
                testCase.verifyEqual(csOutActual(:,idx),csOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caOutExpctd(:,idx) - caOutActual(:,idx))./abs(caOutExpctd(:,idx)));
                testCase.verifyEqual(caOutActual(:,idx),caOutExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
        end
        
    end
    
end
