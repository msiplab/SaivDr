classdef CplxOvsdLpPuFb2dTypeIIVm0SystemTestCase < matlab.unittest.TestCase
    %OVSDLPPUFB2DTYPEIIVM0SYSTEMTESTCASE Test case for CplxOvsdLpPuFb2dTypeIIVm0System
    %
    % SVN identifier:
    % $Id: CplxOvsdLpPuFb2dTypeIIVm0SystemTestCase.m 240 2014-02-23 13:44:58Z sho $
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
        lppufb;
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.lppufb);
        end
    end
    
    methods (Test)
        
        % Test for default construction
        function testConstructor(testCase)
            
            % Expected values
            coefExpctd = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System();
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for default construction
        function testConstructorWithDeepCopy(testCase)
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System();
            cloneLpPuFb = clone(testCase.lppufb);
            
            % Expected values
            coefExpctd = step(testCase.lppufb,[],[]);
            
            % Actual values
            coefActual = step(cloneLpPuFb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
            % Change angles
            angles = randn(size(get(cloneLpPuFb,'Angles')));
            
            % Actual values
            coefActual = step(cloneLpPuFb,angles,[]);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThan(1e-14),sprintf('%g',coefDist));
            
        end
        
%         %TODO: dctmtxをFFT行列に変更する
%         % Test for construction
%         function testConstructorWithDec33Ord00(testCase)
%             
%             % Parameters
%             dec = [ 3 3 ];
%             ord = [ 0 0 ];
%             
%             % Expected values
%             C = dctmtx(3);
%             coefExpctd(1,:,1,1) = reshape(C(1,:).'*C(1,:),1,9);
%             coefExpctd(2,:,1,1) = reshape(C(3,:).'*C(1,:),1,9);
%             coefExpctd(3,:,1,1) = reshape(C(1,:).'*C(3,:),1,9);
%             coefExpctd(4,:,1,1) = reshape(C(3,:).'*C(3,:),1,9);
%             coefExpctd(5,:,1,1) = reshape(C(2,:).'*C(2,:),1,9);
%             coefExpctd(6,:,1,1) = -reshape(C(2,:).'*C(1,:),1,9);
%             coefExpctd(7,:,1,1) = -reshape(C(2,:).'*C(3,:),1,9);
%             coefExpctd(8,:,1,1) = -reshape(C(1,:).'*C(2,:),1,9);
%             coefExpctd(9,:,1,1) = -reshape(C(3,:).'*C(2,:),1,9);
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',dec,...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
        % Test for construction
        function testConstructorWithDec22Ch5Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 0 2
        function testConstructorWithDec22Ch5Ord02Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 2 ];
            ang = 2*pi*rand(1,10+10);
            sym = 2*pi*rand(1,5);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 0
        function testConstructorWithDec22Ch5Ord20Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 2 0 ];
            ang = 2*pi*rand(10+10,1);
            sym = 2*pi*rand(1,5);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch5Ord22Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(1,10+4*5);
            sym = 2*pi*rand(1,5);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));      
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        
%         % Test for construction with order 0 2
%         function testConstructorWithDec33Ord02(testCase)
%             
%             % Parameters
%             dec = [ 3 3 ];
%             ord = [ 0 2 ];
%             ang = 2*pi*rand(16,2);
%             
%             % Expected values
%             nDec = dec(1)*dec(2);
%             nCh = nDec;
%             dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',dec,...
%                 'PolyPhaseOrder',ord);
%                             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));  
%             
%             % Check tightness
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
%                 coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
%             coefDist = max(abs(coefActual(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 2 0
%         function testConstructorWithDec33Ord20(testCase)
%             
%             % Parameters
%             dec = [ 3 3 ];
%             ord = [ 2 0 ];
%             ang = 2*pi*rand(16,2);
%             
%             % Expected values
%             nDec = dec(1)*dec(2);
%             nCh = nDec;
%             dimExpctd = [nCh nDec ord(1)+1 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',dec,...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%             % Check tightness
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
%                 coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
%             coefDist = max(abs(coefActual(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 2 2
%         function testConstructorWithDec33Ord22(testCase)
%             
%             % Parameters
%             dec = [ 3 3 ];
%             ord = [ 2 2 ];
%             ang = 2*pi*rand(16,3);
%             
%             % Expected values
%             nDec = dec(1)*dec(2);
%             nCh = nDec;
%             dimExpctd = [nCh nDec ord(1)+1 ord(2)+1 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',dec,...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist)); 
%             
%             % Check tightness
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
%                 coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
%             coefDist = max(abs(coefActual(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
        
        % Test for construction
        function testConstructorWithDec22Ch7Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 7 ];
            ord = [ 0 0 ];
            
            % Expected values
            dimExpctd = [7 4];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);   
            
            % Check tightness
            import matlab.unittest.constraints.IsLessThan;
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))...
                /sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch9Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 9 ];
            ord = [ 0 0 ];
            
            % Expected values
            dimExpctd = [9 4];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check orthogonality
            import matlab.unittest.constraints.IsLessThan;
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec33Ch9Ord00Ang0(testCase)
            
            % Parameters
            decch = [ 3 3 9 ];
            ord = [ 0 0 ];
            ang = [];
            
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i, 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i;
                 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i , 0.166666666666667 - 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i , 0.166666666666667 - 0.288675134594813i;
                -0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i,-0.166666666666667 - 0.288675134594813i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i;
                 0.166666666666667 + 0.288675134594813i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i ,-0.333333333333333 + 0.00000000000000i,-0.333333333333333 + 0.00000000000000i , 0.166666666666667 - 0.288675134594813i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 - 0.288675134594813i;
                -0.166666666666667 + 0.288675134594813i,-0.166666666666667 - 0.288675134594813i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i, 0.333333333333333 - 0.00000000000000i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 + 0.288675134594813i,-0.166666666666667 - 0.288675134594813i;
                -0.333333333333333 + 0.000000000000000i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 - 0.288675134594813i,-0.333333333333333 + 0.00000000000000i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 - 0.288675134594813i,-0.333333333333333 + 0.00000000000000i;
                -0.166666666666667 + 0.288675134594813i,-0.166666666666667 + 0.288675134594813i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i,-0.166666666666667 - 0.288675134594813i,-0.166666666666667 - 0.288675134594813i;
                -0.333333333333333 + 0.00000000000000i , 0.166666666666667 - 0.288675134594813i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i;
                -0.166666666666667 - 0.288675134594813i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 - 0.00000000000000i ,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i,-0.166666666666667 - 0.288675134594813i, 0.333333333333333 - 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i,-0.166666666666666 + 0.288675134594813i];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec33Ch9Ord00Ang(testCase)
            
            % Parameters
            decch = [ 3 3 9 ];
            ord = [ 0 0 ];
            angV = 2*pi*rand(1,36);
            sym = 2*pi*rand(1,9);
            
            % Expected values
            import saivdr.dictionary.utility.*
            omgsV = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            matrixS = diag(exp(1i*sym));
            matrixV0 = step(omgsV,angV,1);
            coefExpctd(:,:,1,1) = ...
                matrixS*matrixV0 * [
                 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i, 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i;
                 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i , 0.166666666666667 - 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i , 0.166666666666667 - 0.288675134594813i;
                -0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i,-0.166666666666667 - 0.288675134594813i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i;
                 0.166666666666667 + 0.288675134594813i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i ,-0.333333333333333 + 0.00000000000000i,-0.333333333333333 + 0.00000000000000i , 0.166666666666667 - 0.288675134594813i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 - 0.288675134594813i;
                -0.166666666666667 + 0.288675134594813i,-0.166666666666667 - 0.288675134594813i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i, 0.333333333333333 - 0.00000000000000i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 + 0.288675134594813i,-0.166666666666667 - 0.288675134594813i;
                -0.333333333333333 + 0.000000000000000i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 - 0.288675134594813i,-0.333333333333333 + 0.00000000000000i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 - 0.288675134594813i,-0.333333333333333 + 0.00000000000000i;
                -0.166666666666667 + 0.288675134594813i,-0.166666666666667 + 0.288675134594813i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i,-0.166666666666667 - 0.288675134594813i,-0.166666666666667 - 0.288675134594813i;
                -0.333333333333333 + 0.00000000000000i , 0.166666666666667 - 0.288675134594813i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i;
                -0.166666666666667 - 0.288675134594813i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 - 0.00000000000000i ,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i,-0.166666666666667 - 0.288675134594813i, 0.333333333333333 - 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i,-0.166666666666666 + 0.288675134594813i];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
                
            
            % Actual values
            coefActual = step(testCase.lppufb,angV,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-12,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch5Ord00Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 0 ];
            nCh = decch(3);
            ang = 2*pi*rand(1,10);
            sym = 2*pi*rand(1,5);
            
            % Expected values
            dimExpctd = [5 4];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));      
            
            % Check tightness
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch7Ord00Ang(testCase)
            
            % Parameters
            decch = [ 2 2 7 ];
            ord = [ 0 0 ];
            nCh = decch(3);
            ang = 2*pi*rand(1,21);
            sym = 2*pi*rand(1,7);
            
            % Expected values
            dimExpctd = [7 4];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch9Ord00Ang(testCase)
            
            % Parameters
            decch = [ 2 2 9 ];
            ord = [ 0 0 ];
            nCh = decch(3);
            ang = 2*pi*rand(1,36);
            sym = 2*pi*rand(1,9);
            
            % Expected values
            dimExpctd = [9 4];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));     
            
            % Check orthogonality
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec33Ch11Ord00Ang(testCase)
            
            % Parameters
            decch = [ 3 3 11 ];
            ord = [ 0 0 ];
            nCh = decch(3);
            ang = 2*pi*rand(1,55);
            sym = 2*pi*rand(1,11);
            
            % Expected values
            dimExpctd = [11 9];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec11Ch5Ord00(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = [...
                1;
                0;
                0;
                0;
                0];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec11Ch5Ord00Ang0(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 0 0 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                1 ;
                0 ;
                0 ;
                0 ;
                0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        % TODO: テストケースの名称を変更する
        function testConstructorWithDec11Ch5Ord00AngPi3(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 0 0 ];
            angS = pi/3*ones(1,5);
            angV = pi/3*ones(1,10); % TODO: 
            
            % Expected values
            import saivdr.dictionary.utility.*
            omgsV = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            matrixS = diag(exp(1i*angS));
            matrixV0 = step(omgsV,angV,1);
            coefExpctd(:,:,1,1) = ...
                matrixS*matrixV0 * ...
                [ 1 0 0 0 0 ].';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',angS,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,angV,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Invalid arguments
        function testConstructorWithInvalidArguments(testCase)
            
            % Invalid input
            decch = [ 2 2 5 ];
            ord = [ 0 0 ];
            sizeInvalid = [2 2];
            ang = 2*pi*rand(sizeInvalid);
            
            % Expected value
            sizeExpctd = 10;
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Length of angles must be %d',...
                sizeExpctd);
            
            % Instantiation of target class
            try
                import saivdr.dictionary.cnsoltx.*
                testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                    'DecimationFactor',decch(1:2),...
                    'NumberOfChannels',decch(3:end),...
                    'PolyPhaseOrder',ord);
                step(testCase.lppufb,ang,[]);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
        end
        
        % Test for construction
        function testConstructorWithMusPosNeg(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 0 ];
            ang = [];
            mus = [ 1 1 -1 -1 -1 ].';
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                -1i, -1i,  1i,  1i;
                 1 , -1 , -1 ,  1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,mus);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch5Ord22Ang0(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 2 2 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,2,1) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,3,1) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,1,2) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,2,2) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            coefExpctd(:,:,3,2) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,1,3) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,2,3) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,3,3) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4 4
        function testConstructorWithDec22Ch5Ord44Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 4 4 ];
            nCh = decch(3);
            ang = 2*pi*rand(1,10+8*5);
            sym = 2*pi*rand(1,5);
            
            % Expected values
            nDecs = prod(decch(1:2));
            dimExpctd = [decch(3) nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));     
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 3 3 order 2 2
        function testConstructorWithDec33Ch9Ord22Ang0(testCase)
            
            % Parameters
            decch = [ 3 3 9 ];
            ord = [ 2 2 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1,1) = zeros(9);
            coefExpctd(:,:,2,1) = zeros(9);
            coefExpctd(:,:,3,1) = zeros(9);
            coefExpctd(:,:,1,2) = zeros(9);
            coefExpctd(:,:,2,2) = [
                 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i, 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i;
                 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i , 0.166666666666667 - 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i , 0.166666666666667 - 0.288675134594813i;
                -0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i,-0.166666666666667 - 0.288675134594813i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i;
                 0.166666666666667 + 0.288675134594813i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i ,-0.333333333333333 + 0.00000000000000i,-0.333333333333333 + 0.00000000000000i , 0.166666666666667 - 0.288675134594813i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 - 0.288675134594813i;
                -0.166666666666667 + 0.288675134594813i,-0.166666666666667 - 0.288675134594813i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i, 0.333333333333333 - 0.00000000000000i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 + 0.288675134594813i,-0.166666666666667 - 0.288675134594813i;
                -0.333333333333333 + 0.000000000000000i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 - 0.288675134594813i,-0.333333333333333 + 0.00000000000000i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 - 0.288675134594813i,-0.333333333333333 + 0.00000000000000i;
                -0.166666666666667 + 0.288675134594813i,-0.166666666666667 + 0.288675134594813i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i , 0.333333333333333 + 0.00000000000000i, 0.333333333333333 + 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i,-0.166666666666667 - 0.288675134594813i,-0.166666666666667 - 0.288675134594813i;
                -0.333333333333333 + 0.00000000000000i , 0.166666666666667 - 0.288675134594813i, 0.166666666666667 + 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 - 0.288675134594813i, 0.166666666666667 + 0.288675134594813i,-0.333333333333333 + 0.00000000000000i;
                -0.166666666666667 - 0.288675134594813i,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 - 0.00000000000000i ,-0.166666666666667 + 0.288675134594813i, 0.333333333333333 + 0.00000000000000i,-0.166666666666667 - 0.288675134594813i, 0.333333333333333 - 0.00000000000000i ,-0.166666666666667 - 0.288675134594813i,-0.166666666666666 + 0.288675134594813i];
            coefExpctd(:,:,3,2) = zeros(9);
            coefExpctd(:,:,1,3) = zeros(9);
            coefExpctd(:,:,2,3) = zeros(9);
            coefExpctd(:,:,3,3) = zeros(9);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 2 2
        function testConstructorWithDec22Ch7Ord22Ang(testCase)
            
            % Parameters
            decch = [ 2 2 7 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(1,21+4*10);
            sym = 2*pi*rand(1,7);
            
            % Expected values
            nDecs = prod(decch(1:2));
            nCh = decch(3);
            dimExpctd = [nCh nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test: dec 3 3 order 4 4
        function testConstructorWithDec33Ch9Ord44Ang(testCase)
            
            % Parameters
            decch = [ 3 3 9 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(1,36+8*18);
            sym = 2*pi*rand(1,9);
            
            % Expected values
            nDecs = prod(decch(1:2));
            nCh = decch(3);
            dimExpctd = [nCh nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));     
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
            %TODO: stepメソッドの入出力仕様を定める
        % Test for angle setting
        function testSetAngles(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 0 ];
            angPre = [ pi/4 pi/4 pi/4 pi/4 pi/4 pi/4 pi/4 pi/4 pi/4 pi/4 ].';
            angPst = [ 0 0 0 0 0 0 0 0 0 0 ].';
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,angPre,[]);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThanOrEqualTo;
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThanOrEqualTo(1e-14),sprintf('%g',coefDist));
            
            % Actual values
            coefActual = step(testCase.lppufb,angPst,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for mus setting
        function testSetMus(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 0 ];
            ang = [ 0 0 0 0 0 0 0 0 0 0 ].';
            musPre = [ 1 -1  1 -1 1 ].'; %TODO:
            musPst = 1;
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,musPre);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThanOrEqualTo;
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyThat(coefDist,IsGreaterThanOrEqualTo(1e-14),sprintf('%g',coefDist));
            
            % Actual values
            coefActual = step(testCase.lppufb,[],musPst);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for subsref
        function testAnalysisFilterAt(testCase)
            % Expected value
            anFiltExpctd1 = 1/2*[  1   1  ;  1   1  ];
            anFiltExpctd2 = 1/2*[  1i  1i ; -1i -1i ];
            anFiltExpctd3 = 1/2*[  1i -1i ;  1i -1i ];
            anFiltExpctd4 = 1/2*[ -1   1  ;  1  -1  ];
            anFiltExpctd5 = [ 0 0 ; 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'OutputMode','AnalysisFilterAt');
            
            % Actual values
            anFiltActual1 = step(testCase.lppufb,[],[],1);
            anFiltActual2 = step(testCase.lppufb,[],[],2);
            anFiltActual3 = step(testCase.lppufb,[],[],3);
            anFiltActual4 = step(testCase.lppufb,[],[],4);
            anFiltActual5 = step(testCase.lppufb,[],[],5);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan
            dist = norm(anFiltExpctd1(:)-anFiltActual1(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd2(:)-anFiltActual2(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd3(:)-anFiltActual3(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd4(:)-anFiltActual4(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd5(:)-anFiltActual5(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            
        end
        
               
        function testAnalysisFilters(testCase)
            
            % Expected value
            anFiltExpctd1 = 1/2*[  1   1  ;  1   1  ];
            anFiltExpctd2 = 1/2*[  1i  1i ; -1i -1i ];
            anFiltExpctd3 = 1/2*[  1i -1i ;  1i -1i ];
            anFiltExpctd4 = 1/2*[ -1   1  ;  1  -1  ];
            anFiltExpctd5 = [ 0 0 ; 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'OutputMode','AnalysisFilters');
            
            % Actual values
            anFiltsActual = step(testCase.lppufb,[],[]);
            anFiltActual1 = anFiltsActual(:,:,1);
            anFiltActual2 = anFiltsActual(:,:,2);
            anFiltActual3 = anFiltsActual(:,:,3);
            anFiltActual4 = anFiltsActual(:,:,4);
            anFiltActual5 = anFiltsActual(:,:,5);            
            
           % Evaluation
            import matlab.unittest.constraints.IsLessThan
            dist = norm(anFiltExpctd1(:)-anFiltActual1(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd2(:)-anFiltActual2(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd3(:)-anFiltActual3(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd4(:)-anFiltActual4(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd5(:)-anFiltActual5(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            
        end 
        
        % Test dec 2 2 ch 5 order 0 2
        function testConstructorWithDec22Ch5Ord02(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 2 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 1 ch 5 order 0 2
        function testConstructorWithDec11Ch5Ord02(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 0 2 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  ;
                0  ;
                0  ;
                0  ;
                0  ];
            
            coefExpctd(:,:,1,2) = [
                1 ;
                0 ;
                0 ;
                0 ;
                0 ];
            
            coefExpctd(:,:,1,3) = [
                0  ;
                0  ;
                0  ;
                0  ;
                0  ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 1 ch 5 order 2 2
        function testConstructorWithDec11Ch5Ord22(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 2 2 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1,1) = [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2,1) =  [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3,1) = [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2,1) =  [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2,2) = [...
                1;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2,3) =  [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3,1) = [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3,2) =  [...
                0;
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3,3) = [...
                0;
                0;
                0;
                0;
                0];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 0 2
        function testConstructorWithDec22Ch5Ord20(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 2 0 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch5Ord22(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 2 2 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 0 4
        function testConstructorWithDec22Ch5Ord04(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 4 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            coefExpctd(:,:,1,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 0 4
        function testConstructorWithDec22Ch5Ord04Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 0 4 ];
            ang = 2*pi*rand(1,10+4*5);
            sym = 2*pi*rand(1,5);
            
            % Expected values
            nDecs = prod(decch(1:2));
            nCh = decch(3);
            dimExpctd = [nCh nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 4 0
        function testConstructorWithDec22Ch5Ord40(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 4 0 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            coefExpctd(:,:,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4 0
        function testConstructorWithDec22Ord40Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 4 0 ];
            ang = 2*pi*rand(1,10+4*5);
            sym = 2*pi*rand(1,5);
            
            % Expected values
            nDecs = prod(decch(1:2));
            nCh = decch(3);
            dimExpctd = [nCh nDecs ord(1)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 4 4
        function testConstructorWithDec22Ch5Ord44(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 4 4 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            coefExpctd(:,:,4,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
 
        % Test dec 2 2 ch 5 order 6 6
        function testConstructorWithDec22Ch5Ord66Ang(testCase)
            
            % Parameters
            decch = [ 2 2 5 ];
            ord = [ 6 6 ];
            ang = 2*pi*rand(1,10+12*5);
            sym = 2*pi*rand(1,5);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 ch 9 order 2 2
        function testConstructorWithDec22Ch9Ord22Ang(testCase)
            
            % Parameters
            decch = [ 2 2 9 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(1,36+4*18);
            sym = 2*pi*rand(1,9);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 ch 9 order 4 4
        function testConstructorWithDec22Ch9Ord44Ang(testCase)
            
            % Parameters
            decch = [ 2 2 9 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(1,36+8*18);
            sym = 2*pi*rand(1,9);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));     
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 ch 11 order 4 4
        function testConstructorWithDec22Ch11Ord44Ang(testCase)
            
            % Parameters
            decch = [ 2 2 11 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(1,55+8*27);
            sym = 2*pi*rand(1,11);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 3 3 ch 11 order 2 2
        function testConstructorWithDec33Ch11Ord22Ang(testCase)
            
            % Parameters
            decch = [ 3 3 11 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(1,55+4*27);
            sym = 2*pi*rand(1,11);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);            
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 3 3 ch 11 order 4 4
        function testConstructorWithDec33Ch11Ord44Ang(testCase)
            
            % Parameters
            decch = [ 3 3 11 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(1,55+8*27);
            sym = 2*pi*rand(1,11);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
                        
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 1 ch 5 order 2 2
        function testConstructorWithDec11Ch5Ord22Ang(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(1,10+4*5);
            sym = 2*pi*rand(1,5);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
                        
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));     
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 1 ch 5 order 4 4
        function testConstructorWithDec11Ch5Ord44Ang(testCase)
            
            % Parameters
            decch = [ 1 1 5 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(1,10+8*5);
            sym = 2*pi*rand(1,5);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);

            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 1 ch 7 order 4 4
        function testConstructorWithDec11Ch7Ord44Ang(testCase)
            
            % Parameters
            decch = [ 1 1 7 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(1,21+8*10);
            sym = 2*pi*rand(1,7);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);

            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 1 1 ch 7 order 6 6
        function testConstructorWithDec11Ch7Ord66Ang(testCase)
            
            % Parameters
            decch = [ 1 1 7 ];
            ord = [ 6 6 ];
            ang = 2*pi*rand(1,21+12*10);
            sym = 2*pi*rand(1,7);
            
            % Expected values
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec22Ch32Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
%         % Test for construction
%         function testConstructorWithDec22Ch42Ord00(testCase)
%             
%             % Parameters
%             decch = [ 2 2 4 2 ];
%             ord = [ 0 0 ];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
        
        % Test for construction
        function testConstructorWithDec22Ch43Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 4 3 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;
                 0 ,  0 ,  0 ,  0 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
%         % Test for construction
%         function testConstructorWithDec22Ch52Ord00(testCase)
%             
%             % Parameters
%             decch = [ 2 2 5 2 ];
%             ord = [ 0 0 ];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction
%         function testConstructorWithDec22Ch62Ord00(testCase)
%             
%             % Parameters
%             decch = [ 2 2 6 2 ];
%             ord = [ 0 0 ];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
        
        % Test for construction with order 2 2
        function testConstructorWithDec22Ch32Ord22(testCase)
            
            % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 2 2 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
%         % Test for construction with order 2 2
%         function testConstructorWithDec22Ch42Ord22(testCase)
%             
%             % Parameters
%             decch = [ 2 2 4 2 ];
%             ord = [ 2 2 ];
%             ang = [];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,2) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ];
%             
%             coefExpctd(:,:,3,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 2 2
%         function testConstructorWithDec22Ch52Ord22(testCase)
%             
%             % Parameters
%             decch = [ 2 2 5 2 ];
%             ord = [ 2 2 ];
%             ang = [];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,2) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ];
%             
%             coefExpctd(:,:,3,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 2 2
%         function testConstructorWithDec22Ch53Ord22(testCase)
%             
%             % Parameters
%             decch = [ 2 2 5 3 ];
%             ord = [ 2 2 ];
%             ang = [];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,2) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ;
%                 0  0  0  0  ];
%             
%             coefExpctd(:,:,3,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 2 2
%         function testConstructorWithDec22Ch62Ord22(testCase)
%             
%             % Parameters
%             decch = [ 2 2 6 2 ];
%             ord = [ 2 2 ];
%             ang = [];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,2) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ];
%             
%             coefExpctd(:,:,3,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
        
        % Test dec 2 2 order 4 4
        function testConstructorWithDec22Ch32Ord44(testCase)
            
            % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 4 4 ];
            ang = [];
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ;
                 0 ,  0 ,  0 ,  0 ;];
            
            coefExpctd(:,:,4,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end     
        
%         % Test dec 2 2 order 4 4
%         function testConstructorWithDec22Ch42Ord44(testCase)
%             
%             % Parameters
%             decch = [ 2 2 4 2 ];
%             ord = [ 4 4 ];
%             ang = [];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,3) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ];
%             
%             coefExpctd(:,:,4,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,4) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,4) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,4) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,4) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end  
        
        % Test dec 2 2 order 4 4
        function testConstructorWithDec22Ch32Ord22Ang(testCase)
            
          % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 2 2 ];
            ang = 2*pi*rand(1,10+4*5);
            sym = 2*pi*rand(1,5);
            
            % Expected values
            nDec = prod(decch(1:2));
            nCh = sum(decch(3:4));
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 4 4
        function testConstructorWithDec22Ch32Ord44Ang(testCase)
            
            % Parameters
            decch = [ 2 2 3 2 ];
            ord = [ 4 4 ];
            ang = 2*pi*rand(1,10+8*5);
            sym = 2*pi*rand(1,5);
            
            % Expected values
            nDecs = prod(decch(1:2));
            nCh = sum(decch(3:4));
            dimExpctd = [nCh nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'Symmetry',sym,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nCh
                coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);
            end
            coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));    
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]); 
            E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
            coefActual = double(E'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
%         % Test dec 2 2 order 4 4
%         function testConstructorWithDec22Ch42Ord22Ang(testCase)
%             
%           % Parameters
%             decch = [ 2 2 4 2 ];
%             ord = [ 2 2 ];
%             ang = 2*pi*rand(7,3);
%             
%             % Expected values
%             nDec = prod(decch(1:2));
%             nCh = sum(decch(3:4));
%             dimExpctd = [nCh nDec ord(1)+1 ord(2)+1 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%             % Check tightness
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
%                 coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
%             coefDist = max(abs(coefActual(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
%         
%         % Test dec 2 2 order 4 4
%         function testConstructorWithDec22Ch42Ord44Ang(testCase)
%             
%             % Parameters
%             decch = [ 2 2 4 2 ];
%             ord = [ 4 4 ];
%             ang = 2*pi*rand(7,5);
%             
%             % Expected values
%             nDecs = prod(decch(1:2));
%             nCh = sum(decch(3:4));
%             dimExpctd = [nCh nDecs ord(1)+1 ord(2)+1];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%             % Check orthogonality
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
%                 coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
%             coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
       
        
        % Test for ParameterMatrixSet
        function testParameterMatrixSet(testCase)
            
            % Preparation
            mstab = [ 5 5 ];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixContainer
            paramExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(5),1);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end
       
%         % Test for construction
%         function testConstructorWithDec22Ch24Ord00(testCase)
%             
%             % Parameters
%             decch = [ 2 2 2 4 ];
%             ord = [ 0 0 ];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test for construction
%         function testConstructorWithDec22Ch34Ord00(testCase)
%             
%             % Parameters
%             decch = [ 2 2 3 4 ];
%             ord = [ 0 0 ];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 0  0  0  0 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0  ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test for construction
%         function testConstructorWithDec22Ch25Ord00(testCase)
%             
%             % Parameters
%             decch = [ 2 2 2 5 ];
%             ord = [ 0 0 ];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test for construction
%         function testConstructorWithDec22Ch26Ord00(testCase)
%             
%             % Parameters
%             decch = [ 2 2 2 6 ];
%             ord = [ 0 0 ];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 2 2
%         function testConstructorWithDec22Ch23Ord22(testCase)
%             
%             % Parameters
%             decch = [ 2 2 2 3 ];
%             ord = [ 2 2 ];
%             ang = [];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,2) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 2 2
%         function testConstructorWithDec22Ch24Ord22(testCase)
%             
%             % Parameters
%             decch = [ 2 2 2 4 ];
%             ord = [ 2 2 ];
%             ang = [];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,2) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 2 2
%         function testConstructorWithDec22Ch25Ord22(testCase)
%             
%             % Parameters
%             decch = [ 2 2 2 5 ];
%             ord = [ 2 2 ];
%             ang = [];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,2) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test for construction with order 2 2
%         function testConstructorWithDec22Ch35Ord22(testCase)
%             
%             % Parameters
%             decch = [ 2 2 3 5 ];
%             ord = [ 2 2 ];
%             ang = [];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,2) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 0  0  0  0 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0  ];
%             
%             coefExpctd(:,:,3,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
% 
%         % Test for construction with order 2 2
%         function testConstructorWithDec22Ch26Ord22(testCase)
%             
%             % Parameters
%             decch = [ 2 2 2 6 ];
%             ord = [ 2 2 ];
%             ang = [];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,2) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end
%         
%         % Test dec 2 2 order 4 4
%         function testConstructorWithDec22Ch23Ord44(testCase)
%             
%             % Parameters
%             decch = [ 2 2 2 3 ];
%             ord = [ 4 4 ];
%             ang = [];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,3) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,4) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,4) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,4) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,4) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end     
%         
%         % Test dec 2 2 order 4 4
%         function testConstructorWithDec22Ch24Ord44(testCase)
%             
%             % Parameters
%             decch = [ 2 2 2 4 ];
%             ord = [ 4 4 ];
%             ang = [];
%             
%             % Expected values
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,2) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,3) = 1/2 * [
%                 1  1  1  1 ;
%                 1 -1 -1  1 ;
%                 -1  1 -1  1 ;
%                 -1 -1  1  1 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,3) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,1) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,4) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,4) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,4) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,4) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,1,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,2,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,3,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,4,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             coefExpctd(:,:,5,5) = [
%                 0  0  0  0 ;
%                 0  0  0  0 ;                
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ;
%                 0  0  0  0 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
%             testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
%             
%         end  
%         
%         % Test dec 2 2 order 4 4
%         function testConstructorWithDec22Ch23Ord22Ang(testCase)
%             
%           % Parameters
%             decch = [ 2 2 2 3 ];
%             ord = [ 2 2 ];
%             ang = 2*pi*rand(4,3);
%             
%             % Expected values
%             nDec = prod(decch(1:2));
%             nCh = sum(decch(3:4));
%             dimExpctd = [nCh nDec ord(1)+1 ord(2)+1 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist)); 
%             
%             % Check tightness
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
%                 coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
%             coefDist = max(abs(coefActual(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
% 
%         % Test dec 2 2 order 4 4
%         function testConstructorWithDec22Ch23Ord44Ang(testCase)
%             
%             % Parameters
%             decch = [ 2 2 2 3 ];
%             ord = [ 4 4 ];
%             ang = 2*pi*rand(4,5);
%             
%             % Expected values
%             nDecs = prod(decch(1:2));
%             nCh = sum(decch(3:4));
%             dimExpctd = [nCh nDecs ord(1)+1 ord(2)+1];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));  
%             
%             % Check orthogonality
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
%                 coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
%             coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
% 
%         % Test dec 2 2 order 4 4
%         function testConstructorWithDec22Ch24Ord22Ang(testCase)
%             
%           % Parameters
%             decch = [ 2 2 2 4 ];
%             ord = [ 2 2 ];
%             ang = 2*pi*rand(7,3);
%             
%             % Expected values
%             nDec = prod(decch(1:2));
%             nCh = sum(decch(3:4));
%             dimExpctd = [nCh nDec ord(1)+1 ord(2)+1 ];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%             % Check tightness
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) = ...
%                 coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1) - eye(nDec);
%             coefDist = max(abs(coefActual(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
%         
%         % Test dec 2 2 order 4 4
%         function testConstructorWithDec22Ch24Ord44Ang(testCase)
%             
%             % Parameters
%             decch = [ 2 2 2 4 ];
%             ord = [ 4 4 ];
%             ang = 2*pi*rand(7,5);
%             
%             % Expected values
%             nDecs = prod(decch(1:2));
%             nCh = sum(decch(3:4));
%             dimExpctd = [nCh nDecs ord(1)+1 ord(2)+1];
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'DecimationFactor',decch(1:2),...
%                 'NumberOfChannels',decch(3:end),...
%                 'PolyPhaseOrder',ord);
%             
%             % Actual values
%             coefActual = step(testCase.lppufb,ang,[]);
%             
%             % Evaluation
%             testCase.verifySize(coefActual,dimExpctd);
%             
%             % Check symmetry
%             import matlab.unittest.constraints.IsLessThan;
%             coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
%             coefDist = max(abs(coefDiff(:)));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%             % Check orthogonality
%             coefE = step(testCase.lppufb,[],[]); 
%             E = saivdr.dictionary.utility.PolyPhaseMatrix2d(coefE);
%             coefActual = double(E'*E);
%             coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) = ...
%                 coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1) - eye(nDecs);
%             coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
%             testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
%             
%         end
%        
%         function testParameterMatrixSetCh23(testCase)
%             
%             % Preparation
%             nchs  = [ 2 3 ];
%             mstab = [ 2 2 ; 3 3 ];
%             
%             % Expected value
%             import saivdr.dictionary.utility.ParameterMatrixSet
%             paramExpctd = ParameterMatrixSet(...
%                 'MatrixSizeTable',mstab);
%             step(paramExpctd,eye(2),1);
%             step(paramExpctd,eye(3),2);
%             
%             % Instantiation of target class
%             import saivdr.dictionary.cnsoltx.*
%             testCase.lppufb = CplxOvsdLpPuFb2dTypeIIVm0System(...
%                 'OutputMode','ParameterMatrixSet',...
%                 'NumberOfChannels',nchs);
%             
%             % Actual values
%             paramActual = step(testCase.lppufb,[],[]);
%             
%             % Evaluation
%             testCase.verifyEqual(paramExpctd, paramActual);
%             
%         end
        
    end
    
end
