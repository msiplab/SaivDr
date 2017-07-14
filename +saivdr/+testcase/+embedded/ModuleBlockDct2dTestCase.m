classdef ModuleBlockDct2dTestCase < matlab.unittest.TestCase
    %MODULEBLOCKDCT2DTESTCASE Test case for ModuleBlockDct2d
    %
    % SVN identifier:
    % $Id: ModuleBlockDct2dTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        
        function test2x2Dct(testCase)

            srcBlock = [ 1 3 ; 2 4 ];
            
            % Expected values
            csExpctd = [ 5 0 ].';
            caExpctd = [ -1 -2 ].';
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleBlockDct2d();
            
            % Actual values
            [csActual,caActual] = step(testCase.module,srcBlock);
            
            % Evaluation
            diff = max(abs(csExpctd - csActual)./abs(csExpctd));
            testCase.verifyEqual(csActual,csExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caExpctd - caActual)./abs(caExpctd));
            testCase.verifyEqual(caActual,caExpctd,'RelTol',1e-7,sprintf('%g',diff));            
         end
        
        function test2x2DctRandom(testCase)

            dec = 2;
            srcBlock = rand(dec);
            
            % Expected values
            E0 = [ 1  1  1  1;
                   1 -1 -1  1;
                  -1  1 -1  1;
                  -1 -1  1  1 ]/2;
            coefsExpctd = E0*flipud(srcBlock(:));
            csExpctd = coefsExpctd(1:2);
            caExpctd = coefsExpctd(3:4);
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleBlockDct2d();
            
            % Actual values
            [csActual,caActual] = step(testCase.module,srcBlock);
            
            % Evaluation
            diff = max(abs(csExpctd - csActual)./abs(csExpctd));
            testCase.verifyEqual(csActual,csExpctd,'RelTol',1e-7,sprintf('%g',diff));
            diff = max(abs(caExpctd - caActual)./abs(caExpctd));
            testCase.verifyEqual(caActual,caExpctd,'RelTol',1e-7,sprintf('%g',diff));            
        end
        
        function test2x2DctSuccessiveProcessing(testCase)

            dec = 2;
            srcBlock(:,:,1) = [ 1 5 ; 2 6 ];
            srcBlock(:,:,2) = [ 9 3 ; 0 4 ];
            srcBlock(:,:,3) = [ 3 7 ; 4 8 ];
            srcBlock(:,:,4) = [ 1 5 ; 2 6 ];
            nBlks = size(srcBlock,3);
            
            % Expected values
            csExpctd(:,1) = [ 7 0 ].';
            caExpctd(:,1) = [ -1 -4 ].';
            csExpctd(:,2) = [ 8 5 ].';
            caExpctd(:,2) = [ 4 1 ].';
            csExpctd(:,3) = [ 11 0 ].';
            caExpctd(:,3) = [ -1 -4 ].';            
            csExpctd(:,4) = [ 7 0 ].';
            caExpctd(:,4) = [ -1 -4 ].';                        
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleBlockDct2d();
            
            % Actual values
            csActual = zeros(dec*dec/2,nBlks);
            caActual = zeros(dec*dec/2,nBlks);
            for idx = 1:nBlks
                [csActual(:,idx),caActual(:,idx)] = ...
                    step(testCase.module,srcBlock(:,:,idx));
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csExpctd(:,idx) - csActual(:,idx))./abs(csExpctd(:,idx)));
                testCase.verifyEqual(csActual(:,idx),csExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caExpctd(:,idx) - caActual(:,idx))./abs(caExpctd(:,idx)));
                testCase.verifyEqual(caActual(:,idx),caExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
            
        end
         
        function test2x2DctSuccessiveProcessingRandom(testCase)

            dec = 2;
            nBlks = 4;
            E0 = [
                 1  1  1  1;
                 1 -1 -1  1;
                -1  1 -1  1;
                -1 -1  1  1 ]/2;
            srcBlock = zeros(dec,dec,nBlks);
            csExpctd = zeros(dec*dec/2,nBlks);
            caExpctd = zeros(dec*dec/2,nBlks);
            for idx = 1:nBlks
                subBlock = rand(dec);
                srcBlock(:,:,idx) = subBlock;
                % Expected values
                coefsExpctd = E0*flipud(subBlock(:));
                csExpctd(:,idx) = coefsExpctd(1:2);
                caExpctd(:,idx) = coefsExpctd(3:4);
            end
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleBlockDct2d();
            
            % Actual values
            csActual = zeros(dec*dec/2,nBlks);
            caActual = zeros(dec*dec/2,nBlks);
            for idx = 1:nBlks
                [csActual(:,idx),caActual(:,idx)] = ...
                    step(testCase.module,srcBlock(:,:,idx));
            end
            
            % Evaluation
            for idx = 1:nBlks
                diff = max(abs(csExpctd(:,idx) - csActual(:,idx))./abs(csExpctd(:,idx)));
                testCase.verifyEqual(csActual(:,idx),csExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
                diff = max(abs(caExpctd(:,idx) - caActual(:,idx))./abs(caExpctd(:,idx)));
                testCase.verifyEqual(caActual(:,idx),caExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
            end
            
        end        
         
        function test2x2DctCh2plus2(testCase)

            srcBlock = [ 1 3 ; 2 4 ];
            nChs = [ 2 2 ];
            
            % Expected values
            csExpctd = [ 5 0 ].';
            caExpctd = [ -1 -2 ].';
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleBlockDct2d(...
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2));
            
            % Actual values
            [csActual,caActual] = step(testCase.module,srcBlock);
            
            % Evaluation
            testCase.verifySize(csActual,size(csExpctd));
            diff = max(abs(csExpctd - csActual)./abs(csExpctd));
            testCase.verifyEqual(csActual,csExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifySize(caActual,size(caExpctd));
            diff = max(abs(caExpctd - caActual)./abs(caExpctd));
            testCase.verifyEqual(caActual,caExpctd,'RelTol',1e-7,sprintf('%g',diff));            
        end
          
        function test2x2DctCh5plus2(testCase)

            srcBlock = [ 1 3 ; 2 4 ];
            nChs = [ 5 2 ];
            
            % Expected values
            csExpctd = [ 5 0 0 0 0 ].';
            caExpctd = [ -1 -2 ].';
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleBlockDct2d(...
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2));
            
            % Actual values
            [csActual,caActual] = step(testCase.module,srcBlock);
            
            % Evaluation
            testCase.verifySize(csActual,size(csExpctd));
            diff = max(abs(csExpctd - csActual)./abs(csExpctd));
            testCase.verifyEqual(csActual,csExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifySize(caActual,size(caExpctd));
            diff = max(abs(caExpctd - caActual)./abs(caExpctd));
            testCase.verifyEqual(caActual,caExpctd,'RelTol',1e-7,sprintf('%g',diff));            
        end
          
    end
 
end
