classdef ModuleBlockIdct2dTestCase < matlab.unittest.TestCase
    %MODULEBLOCKIDCT2DTESTCASE Test case for ModuleBlockIdct2d
    %
    % SVN identifier:
    % $Id: ModuleBlockIdct2dTestCase.m 683 2015-05-29 08:22:13Z sho $
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
        
        function test2x2Idct(testCase)

            cs = [ 5 0 ].';
            ca = [ -1 -2 ].';
            
            % Expected values
            resExpctd = [ 1 3 ; 2 4 ];
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleBlockIdct2d();
            
            % Actual values
            resActual = step(testCase.module,cs,ca);
            
            % Evaluation
            testCase.verifySize(resActual,size(resExpctd));
            diff = max(abs(resExpctd(:) - resActual(:))./abs(resExpctd(:)));
            testCase.verifyEqual(resActual(:),resExpctd(:),'RelTol',1e-7,sprintf('%g',diff));
         end

        function test2x2IdctRandom(testCase)

            dec = 2;
            cs = rand(dec*dec/2,1);
            ca = rand(dec*dec/2,1);
            
            % Expected values
            E0 = [ 1  1  1  1;
                   1 -1 -1  1;
                  -1  1 -1  1;
                  -1 -1  1  1 ]/2;
            resExpctd = reshape(flipud(E0.'*[cs ; ca ]),dec,dec);
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleBlockIdct2d();
            
            % Actual values
            resActual = step(testCase.module,cs,ca);
            
            % Evaluation
            testCase.verifySize(resActual,size(resExpctd));
            diff = max(abs(resExpctd(:) - resActual(:))./abs(resExpctd(:)));
            testCase.verifyEqual(resActual(:),resExpctd(:),'RelTol',1e-7,sprintf('%g',diff));
        end

        function test2x2IdctSuccessiveProcessing(testCase)

            dec = 2;
            cs(:,1) = [ 7 0 ].';
            ca(:,1) = [ -1 -4 ].';
            cs(:,2) = [ 8 5 ].';
            ca(:,2) = [ 4 1 ].';
            cs(:,3) = [ 11 0 ].';
            ca(:,3) = [ -1 -4 ].';            
            cs(:,4) = [ 7 0 ].';
            ca(:,4) = [ -1 -4 ].';                        
            nBlks = size(cs,2);
            
            % Expected values
            resExpctd(:,:,1) = [ 1 5 ; 2 6 ];
            resExpctd(:,:,2) = [ 9 3 ; 0 4 ];
            resExpctd(:,:,3) = [ 3 7 ; 4 8 ];
            resExpctd(:,:,4) = [ 1 5 ; 2 6 ];        
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleBlockIdct2d();
            
            % Actual values
            resActual = zeros(dec,dec,nBlks);
            for idx = 1:nBlks
                resActual(:,:,idx) = ...
                    step(testCase.module,cs(:,idx),ca(:,idx));
            end
            
            % Evaluation
            diff = max(abs(resExpctd(:) - resActual(:))./abs(resExpctd(:)));
            testCase.verifyEqual(resActual,resExpctd,'RelTol',1e-7,sprintf('%g',diff));
        end
        
        function test2x2IdctSuccessiveProcessingRandom(testCase)

            dec = 2;
            nBlks = 4;
            E0 = [
                 1  1  1  1;
                 1 -1 -1  1;
                -1  1 -1  1;
                -1 -1  1  1 ]/2;
            
            cs = zeros(dec*dec/2,nBlks);
            ca = zeros(dec*dec/2,nBlks);
            resExpctd = zeros(dec,dec,nBlks);
            for idx = 1:nBlks
                cs_ = rand(dec*dec/2,1);
                ca_ = rand(dec*dec/2,1);
                cs(:,idx) = cs_;
                ca(:,idx) = ca_;
                % Expected values
                resExpctd(:,:,idx) = reshape(flipud(E0.')*[cs_ ; ca_],dec,dec);
            end
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleBlockIdct2d();
            
            % Actual values
            resActual = zeros(dec,dec);
            for idx = 1:nBlks
                resActual(:,:,idx) = ...
                    step(testCase.module,cs(:,idx),ca(:,idx));
            end
            
            % Evaluation
            testCase.verifySize(resActual,size(resExpctd));
            diff = max(abs(resExpctd(:) - resActual(:))./abs(resExpctd(:)));
            testCase.verifyEqual(resActual(:,idx),resExpctd(:,idx),'RelTol',1e-7,sprintf('%g',diff));
        end

                
        function test2x2IdctCh2plus2(testCase)

            cs = [ 5 0 ].';
            ca = [ -1 -2 ].';
            %nChs = [ 2 2 ];
            
            % Expected values
            resExpctd = [ 1 3 ; 2 4 ];
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleBlockIdct2d();
            
            % Actual values
            resActual = step(testCase.module,cs,ca);
            
            % Evaluation
            testCase.verifySize(resActual,size(resExpctd));
            diff = max(abs(resExpctd(:) - resActual(:))./abs(resExpctd(:)));
            testCase.verifyEqual(resActual(:),resExpctd(:),'RelTol',1e-7,sprintf('%g',diff));
        end
         
        function test2x2IdctCh5plus2(testCase)

            cs = [ 5 0 0 0 0].';
            ca = [ -1 -2 ].';
            %nChs = [ 5 2 ];
            
            % Expected values
            resExpctd = [ 1 3 ; 2 4 ];
            
            % Instantiation of target class
            import saivdr.embedded.*
            testCase.module = ModuleBlockIdct2d();
            
            % Actual values
            resActual = step(testCase.module,cs,ca);
            
            % Evaluation
            testCase.verifySize(resActual,size(resExpctd));
            diff = max(abs(resExpctd(:) - resActual(:))./abs(resExpctd(:)));
            testCase.verifyEqual(resActual(:),resExpctd(:),'RelTol',1e-7,sprintf('%g',diff));
         end        
        
    end
 
end
