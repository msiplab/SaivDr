classdef OvsdLpPuFb3dTypeIVm0SystemTestCase < matlab.unittest.TestCase
    %OvsdLpPuFb3dTypeIVm0SystemTESTCASE Test case for OvsdLpPuFb3dTypeIVm0System
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2016, Shogo MURAMATSU
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
    
    properties (Constant)
        %{
            F00x = idct2([1 0; 0 0]);
            F10x = idct2([0 0; 1 0]);
            F01x = idct2([0 1; 0 0]);
            F11x = idct2([0 0; 0 1]);
            Fxx0 = permute(idct([1 ; 0]),[2 3 1]);
            Fxx1 = permute(idct([0 ; 1]),[2 3 1]);
            F000 = convn(F00x,Fxx0);
            F001 = convn(F00x,Fxx1);
            F100 = convn(F10x,Fxx0);
            F101 = convn(F10x,Fxx1);
            F010 = convn(F01x,Fxx0);
            F011 = convn(F01x,Fxx1);
            F110 = convn(F11x,Fxx0);
            F111 = convn(F11x,Fxx1);
            matrixE0 = flip([ ...
                F000(:).'
                F011(:).'
                F110(:).'
                F101(:).'
                F001(:).'
                F010(:).'
                F111(:).'
                F100(:).' ],2);
        %}
        matrixE0 = 1/(2*sqrt(2))*[
            1     1     1     1     1     1     1     1
            1     1    -1    -1    -1    -1     1     1
            1    -1    -1     1     1    -1    -1     1
            1    -1     1    -1    -1     1    -1     1
            -1    -1    -1    -1     1     1     1     1
            -1    -1     1     1    -1    -1     1     1
            -1     1     1    -1     1    -1    -1     1
            -1     1    -1     1    -1     1    -1     1
            ];
    end
    
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
            
            % Expected values yxz
            coefExpctd(:,:,1,1,1) = testCase.matrixE0;
        
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System();
            
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
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System();
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
                
        % Test for construction
        function testConstructorWithOrd00(testCase)
            
            % Parameters
            dec = [ 2 2 2 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1,1) = testCase.matrixE0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec444Ord000(testCase)
            
            % Parameters
            dec = [ 4 4 4 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            dimExpctd = [64 64];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithDec222Ch8Ord000(testCase)
            
            % Parameters
            decch = [ 2 2 2 8 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1,1) = testCase.matrixE0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
           coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec222Ch8Ord002(testCase)
            
            % Parameters
            decch = [ 2 2 2 8 ];
            ord = [ 0 0 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1,1) =  [
            ...%yxz
            ...% 000 100 010 110 001 101 011 111
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;                
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0                 
            ];
            coefExpctd(:,:,1,1,2) = testCase.matrixE0;                
            coefExpctd(:,:,1,1,3) =  [
            ...%yxz
            ...% 000 100 010 110 001 101 011 111
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;                
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0                 
            ];
        
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));            
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));            
            
        end
        
        % Test for construction
        function testConstructorWithDec222Ch8Ord020(testCase)
            
            % Parameters
            decch = [ 2 2 2 8 ];
            ord = [ 0 2 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1,1) =  [
            ...%yxz
            ...% 000 100 010 110 001 101 011 111
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;                
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0                 
            ];
            coefExpctd(:,:,1,2,1) = testCase.matrixE0;                        
            coefExpctd(:,:,1,3,1) =  [
            ...%yxz
            ...% 000 100 010 110 001 101 011 111
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;                
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0                 
            ];
        
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));            
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));            
            
        end
        
        % Test for construction
        function testConstructorWithDec222Ch8Ord200(testCase)
            
            % Parameters
            decch = [ 2 2 2 8 ];
            ord = [ 2 0 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1,1) =  [
            ...%yxz
            ...% 000 100 010 110 001 101 011 111
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;                
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0                 
            ];
            coefExpctd(:,:,2,1,1) = testCase.matrixE0;
            coefExpctd(:,:,3,1,1) =  [
            ...%yxz
            ...% 000 100 010 110 001 101 011 111
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0 ;                
                0  0  0  0  0  0  0  0 ;
                0  0  0  0  0  0  0  0                 
            ];
        
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));            
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));            
            
        end
        
        % Test for construction
        function testConstructorWithDec222Ch8Ord022(testCase)
            
            % Parameters
            decch = [ 2 2 2 8 ];
            ord = [ 0 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(8,8,1,3,3);
            coefExpctd(:,:,1,2,2) = testCase.matrixE0;                
        
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));            
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));            
            
        end
             
        % Test for construction
        function testConstructorWithDec222Ch8Ord220(testCase)
            
            % Parameters
            decch = [ 2 2 2 8 ];
            ord = [ 2 2 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(8,8,3,3,1);
            coefExpctd(:,:,2,2,1) = testCase.matrixE0;                            

            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));            
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));            
            
        end
           
        % Test for construction
        function testConstructorWithDec222Ch8Ord202(testCase)
            
            % Parameters
            decch = [ 2 2 2 8 ];
            ord = [ 2 0 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(8,8,3,1,3);
            coefExpctd(:,:,2,1,2) = testCase.matrixE0;                                        
        
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));            
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));            
            
        end
           
        % Test for construction
        function testConstructorWithDec222Ch8Ord222(testCase)
            
            % Parameters
            decch = [ 2 2 2 8 ];
            ord = [ 2 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(8,8,3,3,3);
            coefExpctd(:,:,2,2,2) = testCase.matrixE0;                                                    
        
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));            
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));            
            
        end
        
        %{
        % Test for construction
        function testConstructorWithDec222Ch8Ord001(testCase)
            
            % Parameters
            decch = [ 2 2 2 8 ];
            ord = [ 0 0 1 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1,1) =  1/(2*sqrt(2))*[
            ...%yxz
            ...% 000 100 010 110 001 101 011 111
                0  0  0  0  1  1  1  1 ;
                0  0  0  0  1  1 -1 -1 ; 
                0  0  0  0  1 -1 -1  1 ;
                0  0  0  0  1 -1  1 -1 ;
                0  0  0  0  1  1  1  1 ;
                0  0  0  0  1  1 -1 -1 ;
                0  0  0  0  1 -1 -1  1 ;
                0  0  0  0  1 -1  1 -1 
            ];
            coefExpctd(:,:,1,1,2) =  1/(2*sqrt(2))*[
                1  1  1  1  0  0  0  0 ;
               -1 -1  1  1  0  0  0  0 ;
                1 -1 -1  1  0  0  0  0 ;
               -1  1 -1  1  0  0  0  0 ;
               -1 -1 -1 -1  0  0  0  0 ;
                1  1 -1 -1  0  0  0  0 ;
               -1  1  1 -1  0  0  0  0 ;
                1 -1  1 -1  0  0  0  0 
            ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));            
            
        end
        %}
        
        %{
        % Test for construction
        function testConstructorWithDec222Ch8Ord010(testCase)
            
            % Parameters
            decch = [ 2 2 2 8 ];
            ord = [ 0 1 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1,1) =  1/(2*sqrt(2))*[
            ...%yxz
            ...% 000 100 010 110 001 101 011 111
                0  0  1  1  0  0  1  1  ;
                0  0  1  1  0  0 -1 -1  ;
                0  0  1 -1  0  0  1 -1  ;
                0  0  1 -1  0  0 -1  1  ;
                0  0  1  1  0  0 -1 -1  ;
                0  0  1  1  0  0  1  1  ;
                0  0  1 -1  0  0 -1  1  ;
                0  0  1 -1  0  0  1 -1  
            ];
            coefExpctd(:,:,1,2,1) =  1/(2*sqrt(2))*[
                1  1  0  0  1  1  0  0 ;                
               -1 -1  0  0  1  1  0  0 ;
               -1  1  0  0 -1  1  0  0 ;
                1 -1  0  0 -1  1  0  0 ;
                1  1  0  0 -1 -1  0  0 ;
               -1 -1  0  0 -1 -1  0  0 ;
               -1  1  0  0  1 -1  0  0 ;
                1 -1  0  0  1 -1  0  0 
            ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));            
            
        end
        %}
        
        %TODO function testConstructorWithDec222Ch8Ord100(testCase)
        %end
        
        %TODO function testConstructorWithDec222Ch8Ord011(testCase)
        %end        
        
        %TODO function testConstructorWithDec222Ch8Ord110(testCase)
        %end                
        
        %TODO function testConstructorWithDec222Ch8Ord111(testCase)
        %end                        
        
        % Test for construction with order 2 0 0
        function testConstructorWithDec222Ch10Ord200(testCase)
            
            % Parameters
            decch = [ 2 2 2 10 ];
            ord = [ 2 0 0 ];
            ang = 2*pi*rand(10,4);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction with order 0 0 2
        function testConstructorWithDec222Ch10Ord002(testCase)
            
            % Parameters
            decch = [ 2 2 2 10 ];
            ord = [ 0 0 2 ];
            ang = 2*pi*rand(10,4);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
       
        % Test for construction with order 2 0 0
        function testConstructorWithDec222Ch12Ord200(testCase)
            
            % Parameters
            decch = [ 2 2 2 12 ];
            ord = [ 2 0 0 ];
            ang = 2*pi*rand(15,4);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);    
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction with order 0 2 0
        function testConstructorWithDec222Ch12Ord020(testCase)
            
            % Parameters
            decch = [ 2 2 2 12 ];
            ord = [ 0 2 0];
            ang = 2*pi*rand(15,4);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 0 2 0
        function testConstructorWithDec222Ch12Ord002(testCase)
            
            % Parameters
            decch = [ 2 2 2 12 ];
            ord = [ 0 0 2 ];
            ang = 2*pi*rand(15,4);
            
            % Expected values
            nChs = decch(4);
            nDec = prod(decch(1:3));
            dimExpctd = [nChs nDec ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check tightness
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDec,1:nDec,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDec);
            coefDist = max(abs(coefActual(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end        
        
        % Test for construction
        function testConstructorWithDec222Ch12Ord000(testCase)
            
            % Parameters
            decch = [ 2 2 2 12 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            dimExpctd = [ 12 8 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec222Ch10Ord000(testCase)
            
            % Parameters
            decch = [ 2 2 2 10 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            dimExpctd = [ 10 8 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithOrd000Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 ];
            ord = [ 0 0 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1,1) = testCase.matrixE0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithAng0Pi4(testCase)
            
            % Parameters
            decch = [ 2 2 2 ];
            ord = [ 0 0 0 ];
            angW0 = [ 0 0 0 0 0 0 ].';
            angU0 = pi/4 * [ 1 1 1 1 1 1 ].';
            ang   = [ angW0 angU0 ];
            
            % Expected values
            import saivdr.dictionary.utility.*            
            omgs = OrthonormalMatrixGenerationSystem();
            matrixW0 = step(omgs,angW0,1);
            matrixU0 = step(omgs,angU0,1);
            coefExpctd(:,:,1,1,1) = ...
                blkdiag(matrixW0, matrixU0) * testCase.matrixE0;   
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec444Ord000Ang(testCase)
            
            % Parameters
            dec = [ 4 4 4 ];
            ord = [ 0 0 0 ];
            ang = 2*pi*rand(496,2);
            
            % Expected values
            dimExpctd = [64 64];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*          
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec222Ch10Ord000Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 10 ];
            ord = [ 0 0 0 ];
            ang = 2*pi*rand(10,2);
            
            % Expected values
            dimExpctd = [10 8];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);

            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec222Ch12Ord000Ang(testCase)
            
            % Parameters
            decch = [ 2 2 2 12 ];
            ord = [ 0 0 0 ];
            ang = 2*pi*rand(15,2);
            
            % Expected values
            dimExpctd = [12 8];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check orthogonality
            coefDist = norm((coefActual.'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec111Ch8Ord000(testCase)
            
            % Parameters
            decch = [ 1 1 1 8 ];
            ord = [ 0 0 0 ];
            
           % Expected values
            coefExpctd(:,:,1,1,1) = [...
                1;
                0;
                0;
                0;
                0;
                0;
                0;
                0];
                    
             % Instantiation of target class
            import saivdr.dictionary.nsoltx.*             
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec111Ch8Ord000Ang(testCase)
            
            % Parameters
            decch = [ 1 1 1 8 ];
            ord = [ 0 0 0 ];
            ang = zeros(6,2);
            
            % Expected values
            coefExpctd(:,:,1,1,1) = [
                1 ;
                0 ;
                0 ;
                0 ;
                0 ;
                0 ;
                0 ;
                0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
            
        end

         % Test for construction
        function testConstructorWithDec111Ch8Ord00Ang0Pi4(testCase)
            
            % Parameters
            decch = [ 1 1 1 8 ];
            ord = [ 0 0 0 ];
            angW0 = zeros(6,1);
            angU0 = pi/4*ones(6,1);
            ang = [angW0 angU0];
            
            % Expected values
            import saivdr.dictionary.utility.*            
            omgs = OrthonormalMatrixGenerationSystem();
            matrixW0 = step(omgs,angW0,1);
            matrixU0 = step(omgs,angU0,1);
            coefExpctd(:,:,1,1,1) = ...
                blkdiag(matrixW0, matrixU0) * ...
                [ 1 0 0 0 0 0 0 0].';
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        function testConstructorWithInvalidArguments(testCase)
            
            % Invalid input
            dec = [ 2 2 2 ];
            ord = [ 0 0 0 ];
            sizeInvalid = [ 2 2 ];
            ang = 2*pi*rand(sizeInvalid);
            
            % Expected value
            sizeExpctd = [6 2];
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Size of angles must be [ %d %d ]',...
                sizeExpctd(1), sizeExpctd(2));
            
            % Instantiation of target class
            try
                import saivdr.dictionary.nsoltx.*
                OvsdLpPuFb3dTypeIVm0System(...
                    'DecimationFactor',dec,...
                    'PolyPhaseOrder',ord,...
                    'Angles',ang);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
        end

        function testConstructorWithOddChannels(testCase)
            
            % Invalid input
            decch = [ 2 2 2 9 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = '#Channels must be even.';
            
            % Instantiation of target class
            try
                import saivdr.dictionary.nsoltx.*
                OvsdLpPuFb3dTypeIVm0System(...
                    'DecimationFactor',decch(1:3),...
                    'NumberOfChannels',decch(4:end),...
                    'PolyPhaseOrder',ord);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
        end

        function testConstructorWithUnEqualNsNa(testCase)
            
            % Invalid input
            decch = [ 2 2 2 6 2 ];
            ord = [ 0 0 0 ];
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = 'Both of NumberOfSymmetricChannels and NumberOfAntisymmetric channels shoud be greater than or equal to prod(DecimationFactor)/2.';
            
            % Instantiation of target class
            try
                import saivdr.dictionary.nsoltx.*
                OvsdLpPuFb3dTypeIVm0System(...
                    'DecimationFactor',decch(1:3),...
                    'NumberOfChannels',decch(4:end),...
                    'PolyPhaseOrder',ord);
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
            dec = [ 2 2 2 ];
            ord = [ 0 0 0 ];
            ang = zeros(6,2);
            mus = [ones(4,1) -ones(4,1)];
            
            % Expected values
            coefExpctd(:,:,1,1,1) = ...
                blkdiag(eye(4),-eye(4))*testCase.matrixE0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Angles',ang,...
                'Mus',mus);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 2 2
        function testConstructorWithOrd222(testCase)
            
            % Parameters
            dec = [ 2 2 2 ];
            ord = [ 2 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(8,8,3,3,3);
            coefExpctd(:,:,2,2,2) = testCase.matrixE0;
           
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4 4 4
        function testConstructorWithOrd444(testCase)
            
            % Parameters
            dec = [ 2 2 2 ];
            ord = [ 4 4 4 ];
            ang = 2*pi*rand(6,14);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 4 4 order 2 2 2
        function testConstructorWithDec444Ord222(testCase)
            
            % Parameters
            dec = [ 4 4 4 ];
            ord = [ 2 2 2 ];
            ang = 0;
            
            % Expected values
            coefPosExpctd = zeros(64,64,3,3,3);
            coefPosExpctd(:,:,2,2,2) = ones(64);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefPosExpctd));
            testCase.verifyEqual(double(abs(coefActual)>0),coefPosExpctd);
            
        end
        
        % Test dec 4 4 4 order 2 2 2
        function testConstructorWithDec444Ord222Ang(testCase)
            
            % Parameters
            dec = [ 4 4 4 ];
            ord = [ 2 2 2 ];
            ang = 2*pi*rand(496,8);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1 ord(3)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test: dec 4 4 order 4 4
        function testConstructorWithDec444Ord444Ang(testCase)
            
            % Parameters
            dec = [ 4 4 4 ];
            ord = [ 4 4 4 ];
            ang = 2*pi*rand(496,14);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1 ord(3)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefEvn = coefActual(1:ceil(end/2),:);
            coefDiff = coefEvn-fliplr(coefEvn);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));            
            coefOdd = coefActual(ceil(end/2)+1:end,:);
            coefDiff = coefOdd+fliplr(coefOdd);
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                        
            
            % Check orthogonality
            coefE = step(testCase.lppufb,[],[]);
            E = saivdr.dictionary.utility.PolyPhaseMatrix3d(coefE);
            coefActual = double(E.'*E);
            coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1,ord(3)+1) = ...
                coefActual(1:nDecs,1:nDecs,ord(1)+1,ord(2)+1,ord(3)+1) - eye(nDecs);
            coefDist = norm(coefActual(:))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end

        % Test for angle setting
        function testSetAngles(testCase)
            
            % Parameters
            dec = [ 2 2 2 ];
            ord = [ 0 0 0 ];
            angPre = pi/4*ones(6,2);
            angPst = zeros(6,2);
            
            % Expected values
            coefExpctd(:,:,1,1,1) = testCase.matrixE0;

            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',dec,...
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

        % Test for angle setting
        function testSetMus(testCase)
            
            % Parameters
            dec = [ 2 2 2 ];
            ord = [ 0 0 0 ];
            ang = zeros(6,2);
            musPre = diag([1,-1,1,-1])*ones(4,2);
            musPst = 1;
            
            % Expected values
            coefExpctd(:,:,1,1,1) = testCase.matrixE0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',dec,...
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
            anFiltExpctd1(:,:,1) = 1/(2*sqrt(2)) * [
                1  1 ;
                1  1
                ]; 
            anFiltExpctd1(:,:,2) = 1/(2*sqrt(2)) * [
                1  1 ; 
                1  1
                ];             
            anFiltExpctd2(:,:,1) = 1/(2*sqrt(2)) * [
                1 -1 ;
                1 -1
                ];                
            anFiltExpctd2(:,:,2) = 1/(2*sqrt(2)) * [
               -1  1 ;  
               -1  1 
                ];                            
            anFiltExpctd3(:,:,1) = 1/(2*sqrt(2)) * [
                1 -1
               -1  1
                ];                
            anFiltExpctd3(:,:,2) = 1/(2*sqrt(2)) * [
                1 -1
               -1  1
                ];                            
            anFiltExpctd4(:,:,1) = 1/(2*sqrt(2)) * [
                1  1
               -1 -1
                ];                
            anFiltExpctd4(:,:,2) = 1/(2*sqrt(2)) * [
               -1 -1
                1  1
                ];                            
            anFiltExpctd5(:,:,1) = 1/(2*sqrt(2)) * [
               -1 -1
               -1 -1
                ];                
            anFiltExpctd5(:,:,2) = 1/(2*sqrt(2)) * [
                1  1
                1  1 
                ];                            
            anFiltExpctd6(:,:,1) = 1/(2*sqrt(2)) * [
               -1  1
               -1  1
                ];                
            anFiltExpctd6(:,:,2) = 1/(2*sqrt(2)) * [
               -1  1
               -1  1 
                ];                            
            anFiltExpctd7(:,:,1) = 1/(2*sqrt(2)) * [
               -1  1
                1 -1
                ];                
            anFiltExpctd7(:,:,2) = 1/(2*sqrt(2)) * [
                1 -1
               -1  1 
                ];                
            anFiltExpctd8(:,:,1) = 1/(2*sqrt(2)) * [
               -1 -1
                1  1
                ];
            anFiltExpctd8(:,:,2) = 1/(2*sqrt(2)) * [
               -1 -1
                1  1
                ];            
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'OutputMode','AnalysisFilterAt');
            
            % Actual values
            anFiltActual1 = step(testCase.lppufb,[],[],1);
            anFiltActual2 = step(testCase.lppufb,[],[],2);
            anFiltActual3 = step(testCase.lppufb,[],[],3);
            anFiltActual4 = step(testCase.lppufb,[],[],4);
            anFiltActual5 = step(testCase.lppufb,[],[],5);
            anFiltActual6 = step(testCase.lppufb,[],[],6);
            anFiltActual7 = step(testCase.lppufb,[],[],7);
            anFiltActual8 = step(testCase.lppufb,[],[],8);            
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            testCase.verifySize(anFiltActual1,size(anFiltExpctd1));
            testCase.verifySize(anFiltActual2,size(anFiltExpctd2));            
            testCase.verifySize(anFiltActual3,size(anFiltExpctd3));
            testCase.verifySize(anFiltActual4,size(anFiltExpctd4));                        
            testCase.verifySize(anFiltActual5,size(anFiltExpctd5));
            testCase.verifySize(anFiltActual6,size(anFiltExpctd6));            
            testCase.verifySize(anFiltActual7,size(anFiltExpctd7));
            testCase.verifySize(anFiltActual8,size(anFiltExpctd8));                                    
            
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
            dist = norm(anFiltExpctd6(:)-anFiltActual6(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd7(:)-anFiltActual7(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd8(:)-anFiltActual8(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));            
            
        end
        
        function testAnalysisFilters(testCase)
            
            % Expected value
            anFiltExpctd1(:,:,1) = 1/(2*sqrt(2)) * [
                1  1 ;
                1  1
                ]; 
            anFiltExpctd1(:,:,2) = 1/(2*sqrt(2)) * [
                1  1 ; 
                1  1
                ];             
            anFiltExpctd2(:,:,1) = 1/(2*sqrt(2)) * [
                1 -1 ;
                1 -1
                ];                
            anFiltExpctd2(:,:,2) = 1/(2*sqrt(2)) * [
               -1  1 ;  
               -1  1 
                ];                            
            anFiltExpctd3(:,:,1) = 1/(2*sqrt(2)) * [
                1 -1
               -1  1
                ];                
            anFiltExpctd3(:,:,2) = 1/(2*sqrt(2)) * [
                1 -1
               -1  1
                ];                            
            anFiltExpctd4(:,:,1) = 1/(2*sqrt(2)) * [
                1  1
               -1 -1
                ];                
            anFiltExpctd4(:,:,2) = 1/(2*sqrt(2)) * [
               -1 -1
                1  1
                ];                            
            anFiltExpctd5(:,:,1) = 1/(2*sqrt(2)) * [
               -1 -1
               -1 -1
                ];                
            anFiltExpctd5(:,:,2) = 1/(2*sqrt(2)) * [
                1  1
                1  1 
                ];                            
            anFiltExpctd6(:,:,1) = 1/(2*sqrt(2)) * [
               -1  1
               -1  1
                ];                
            anFiltExpctd6(:,:,2) = 1/(2*sqrt(2)) * [
               -1  1
               -1  1 
                ];                            
            anFiltExpctd7(:,:,1) = 1/(2*sqrt(2)) * [
               -1  1
                1 -1
                ];                
            anFiltExpctd7(:,:,2) = 1/(2*sqrt(2)) * [
                1 -1
               -1  1 
                ];                
            anFiltExpctd8(:,:,1) = 1/(2*sqrt(2)) * [
               -1 -1
                1  1
                ];
            anFiltExpctd8(:,:,2) = 1/(2*sqrt(2)) * [
               -1 -1
                1  1
                ];    
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'OutputMode','AnalysisFilters');
            
            % Actual values
            anFiltsActual = step(testCase.lppufb,[],[]);
            anFiltActual1 = anFiltsActual(:,:,:,1);
            anFiltActual2 = anFiltsActual(:,:,:,2);
            anFiltActual3 = anFiltsActual(:,:,:,3);
            anFiltActual4 = anFiltsActual(:,:,:,4);
            anFiltActual5 = anFiltsActual(:,:,:,5);
            anFiltActual6 = anFiltsActual(:,:,:,6);
            anFiltActual7 = anFiltsActual(:,:,:,7);
            anFiltActual8 = anFiltsActual(:,:,:,8);            
            
            % Evaluation
            testCase.verifySize(anFiltActual1,size(anFiltExpctd1));
            testCase.verifySize(anFiltActual2,size(anFiltExpctd2));            
            testCase.verifySize(anFiltActual3,size(anFiltExpctd3));
            testCase.verifySize(anFiltActual4,size(anFiltExpctd4));                        
            testCase.verifySize(anFiltActual5,size(anFiltExpctd5));
            testCase.verifySize(anFiltActual6,size(anFiltExpctd6));            
            testCase.verifySize(anFiltActual7,size(anFiltExpctd7));
            testCase.verifySize(anFiltActual8,size(anFiltExpctd8));                                    
            
            dist = max(abs(anFiltExpctd1(:)-anFiltActual1(:))./abs(anFiltExpctd1(:)));
            testCase.verifyEqual(anFiltActual1,anFiltExpctd1,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd2(:)-anFiltActual2(:))./abs(anFiltExpctd2(:)));
            testCase.verifyEqual(anFiltActual2,anFiltExpctd2,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd3(:)-anFiltActual3(:))./abs(anFiltExpctd3(:)));
            testCase.verifyEqual(anFiltActual3,anFiltExpctd3,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd4(:)-anFiltActual4(:))./abs(anFiltExpctd4(:)));
            testCase.verifyEqual(anFiltActual4,anFiltExpctd4,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd5(:)-anFiltActual5(:))./abs(anFiltExpctd5(:)));
            testCase.verifyEqual(anFiltActual5,anFiltExpctd5,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd6(:)-anFiltActual6(:))./abs(anFiltExpctd6(:)));
            testCase.verifyEqual(anFiltActual6,anFiltExpctd6,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd7(:)-anFiltActual7(:))./abs(anFiltExpctd7(:)));
            testCase.verifyEqual(anFiltActual7,anFiltExpctd7,'RelTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd8(:)-anFiltActual8(:))./abs(anFiltExpctd8(:)));
            testCase.verifyEqual(anFiltActual8,anFiltExpctd8,'RelTol',1e-15,sprintf('%g',dist));
             
        end
 
        % Test dec 2 2 order 0 2
        function testConstructorWithDec222Ord002(testCase)
            
            % Parameters
            dec = [ 2 2 2 ];
            ord = [ 0 0 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd = zeros(8,8,1,1,3);
            coefExpctd(:,:,1,1,2) =  testCase.matrixE0;
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*            
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test dec 1 1 1 4 order 0 0 2
        function testConstructorWithDec111Ch4Ord002(testCase)
            
            % Parameters
            decch = [ 1 1 1 4 ];
            ord = [ 0 0 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1,1) = [
                0  ;
                0  ;
                0  ;
                0  ];
            
            coefExpctd(:,:,1,1,2) = [
                1 ;
                0 ;
                0 ;
                0 ];
            
            coefExpctd(:,:,1,1,3) = [
                0  ;
                0  ;
                0  ;
                0  ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',decch(1:3),...
                'NumberOfChannels',decch(4:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end     
        
        % Test for ParameterMatrixSet
        function testParameterMatrixContainer(testCase)
            
            % Preparation
            mstab = [ 4 4 ; 4 4 ];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixContainer
            paramExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(4),1);
            step(paramExpctd,eye(4),2);
            
            % Instantiation of target class
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end

        function testStepOrd000Ch44Rand(testCase)

            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            omg = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            ord  = [ 0 0 0 ];
            nch  = [ 4 4 ];
            npm = sum((nch(1)-1:1));
            angs = rand(npm,2);
            mus  = ones(nch(1),2);
            W0 = step(omg,angs(:,1),mus(:,1));
            U0 = step(omg,angs(:,2),mus(:,2));
            
            % Instantiation
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',[ 2 2 2 ],...
                'NumberOfChannels',[ 4 4 ],...
                'PolyPhaseOrder',ord,...
                'OutputMode','Coefficients');            
            set(testCase.lppufb,'Angles',angs);
            set(testCase.lppufb,'Mus',mus);

            % Expected values
            import saivdr.dictionary.utility.PolyPhaseMatrix3d
            E0 = PolyPhaseMatrix3d(testCase.matrixE0);
            G0 = PolyPhaseMatrix3d(blkdiag(W0,U0));
            E = G0*E0;
            
            % Actual values
            ordExpctd = ord;
            cfsExpctd = double(E);
                                    
            ordActual = get(testCase.lppufb,'PolyPhaseOrder');
            cfsActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8);
            
        end                 

        function testStepOrd222Ch44Rand(testCase)

            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            omg = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            ord  = [ 2 2 2 ];
            nch  = [ 4 4 ];
            npm = 6;
            angs = rand(npm,2+sum(ord));
            mus  = ones(nch(1),2+sum(ord));
            I   = eye(nch(1));
            Z   = zeros(nch(1));
            Dz = zeros(8,8,1,1,2);
            Dz(:,:,1,1,1) = [ I Z ; Z Z ];
            Dz(:,:,1,1,2) = [ Z Z ; Z I ];
            Dx = zeros(8,8,1,2,1);
            Dx(:,:,1,1,1) = [ I Z ; Z Z ];
            Dx(:,:,1,2,1) = [ Z Z ; Z I ];
            Dy = zeros(8,8,2,1,1);
            Dy(:,:,1,1,1) = [ I Z ; Z Z ];
            Dy(:,:,2,1,1) = [ Z Z ; Z I ];            
            W0  = step(omg,angs(:,1),mus(:,1));
            U0  = step(omg,angs(:,2),mus(:,2));
            Uz1 = step(omg,angs(:,3),mus(:,3));
            Uz2 = step(omg,angs(:,4),mus(:,4));            
            Ux1 = step(omg,angs(:,5),mus(:,5));
            Ux2 = step(omg,angs(:,6),mus(:,6));           
            Uy1 = step(omg,angs(:,7),mus(:,7));
            Uy2 = step(omg,angs(:,8),mus(:,8));
            B  = [ I I ; I -I ]/sqrt(2);
            Qz = B*PolyPhaseMatrix3d(Dz)*B;
            Qx = B*PolyPhaseMatrix3d(Dx)*B;
            Qy = B*PolyPhaseMatrix3d(Dy)*B;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.*
            testCase.lppufb = OvsdLpPuFb3dTypeIVm0System(...
                'DecimationFactor',[ 2 2 2 ],...
                'NumberOfChannels',[ 4 4 ],...
                'PolyPhaseOrder',ord,...
                'OutputMode','Coefficients');            
            set(testCase.lppufb,'Angles',angs);
            set(testCase.lppufb,'Mus',mus);

            % Expected values
            import saivdr.dictionary.utility.PolyPhaseMatrix3d
            E0 = testCase.matrixE0;
            R0 = blkdiag(W0,U0);
            Rz1 = blkdiag(I,Uz1);
            Rz2 = blkdiag(I,Uz2);
            Rx1 = blkdiag(I,Ux1);
            Rx2 = blkdiag(I,Ux2);
            Ry1 = blkdiag(I,Uy1);
            Ry2 = blkdiag(I,Uy2);
            E = Ry2*Qy*Ry1*Qy*Rx2*Qx*Rx1*Qx*Rz2*Qz*Rz1*Qz*R0*E0;
            
            % Actual values
            ordExpctd = ord;
            cfsExpctd = E.Coefficients;
                                    
            ordActual = get(testCase.lppufb,'PolyPhaseOrder');
            cfsActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8);
            
        end                         
  
    end
    
end
