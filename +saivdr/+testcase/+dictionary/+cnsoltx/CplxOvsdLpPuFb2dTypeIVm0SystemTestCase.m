classdef CplxOvsdLpPuFb2dTypeIVm0SystemTestCase < matlab.unittest.TestCase
    %OvsdLpPuFb2dTypeIVm0SystemTESTCASE Test case for CplxOvsdLpPuFb2dTypeIVm0System
    %
    % SVN identifier:
    % $Id: CplxOvsdLpPuFb2dTypeIVm0SystemTestCase.m 110 2014-01-16 06:49:46Z sho $
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
                -1 ,  1 ,  1 , -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System();
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for default construction
        function testConstructorWithDeepCopy(testCase)
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System();
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
            dec = [ 2 2 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2*[...
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
           coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec44Ord00(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 0 0 ];
            
            % Expected values
            dimExpctd = [16 16];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                         
            
            % Check orthogonality
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
       
        % Test for construction
        function testConstructorWithDec22Ch4Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 4 ];
            ord = [ 0 0 ];
            
            % Expected values
            coefExpctd = 1/2*[...
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end


         % Test for construction with order 2 0
        function testConstructorWithDec22Ch6Ord20(testCase)
            
            % Parameters
            decch = [ 2 2 6 ];
            ord = [ 2 0 ];
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nCh nDec ord(1)+1]; 
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
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

        % Test for construction with order 0 2
        function testConstructorWithDec22Ch6Ord02(testCase)
            
            % Parameters
            decch = [ 2 2 6 ];
            ord = [ 0 2 ];
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
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
        function testConstructorWithDec22Ch8Ord20(testCase)
            
            % Parameters
            decch = [ 2 2 8 ];
            ord = [ 2 0 ];
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nCh nDec ord(1)+1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
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
        
          % Test for construction with order 0 2
        function testConstructorWithDec22Ch8Ord02(testCase)
            
            % Parameters
            decch = [ 2 2 8 ];
            ord = [ 0 2 ];
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
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
        function testConstructorWithDec22Ch8Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 8 ];
            ord = [ 0 0 ];
            
            % Expected values
            dimExpctd = [8 4];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                      
            
            % Check orthogonality
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
       
       
        % Test for construction
        function testConstructorWithDec22Ch6Ord00(testCase)
            
            % Parameters
            decch = [ 2 2 6 ];
            ord = [ 0 0 ];
            
            % Expected values
            dimExpctd = [6 4];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefDiff = coefActual(:,:)-fliplr(conj(coefActual(:,:)));
            coefDist = max(abs(coefDiff(:)));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));                       
            
            % Check orthogonality
            coefDist = norm((coefActual'*coefActual)-eye(dimExpctd(2)))/sqrt(numel(coefActual));
            testCase.verifyThat(coefDist,IsLessThan(1e-14),sprintf('%g',coefDist));
            
        end
        
        % Test for construction
        function testConstructorWithOrd00Ang(testCase)
            
            % Parameters
            decch = [ 2 2 ];
            ord = [ 0 0 ];
            ang = [ 0 0 0 0 0 0 ];
            
            % Expected values
            coefExpctd = 1/2*[...
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            decch = [ 2 2 ];
            ord = [ 0 0 ];
            ang = [ 0 0 0 0 0 pi/2 ];
            
            % Expected values
            import saivdr.dictionary.utility.*            
            omgs = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            matrixV0 = step(omgs,ang,1);
            coefExpctd(:,:,1,1) = 1/2 * ...
                matrixV0*[...
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction
        function testConstructorWithDec44Ord00Ang(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 0 0 ];
            nCh = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [16 16];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*          
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
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
        function testConstructorWithDec22Ch8Ord00Ang(testCase)
            
            % Parameters
            decch = [ 2 2 8 ];
            ord = [ 0 0 ];
            nCh = decch(3);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [8 4];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);

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
        function testConstructorWithDec22Ch6Ord00Ang(testCase)
            
            % Parameters
            decch = [ 2 2 6 ];
            ord = [ 0 0 ];
            nCh = decch(3);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [6 4];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
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
        function testConstructorWithDec11Ch4Ord00(testCase)
            
            % Parameters
            decch = [ 1 1 4 ];
            ord = [ 0 0 ];
            
           % Expected values
            coefExpctd(:,:,1,1) = [...
                1;
                0;
                0;
                0];
                    
             % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*             
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
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
        function testConstructorWithDec11Ch4Ord00Ang(testCase)
            
            % Parameters
            decch = [ 1 1 4 ];
            ord = [ 0 0 ];
            ang = [ 0 0 0 0 0 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                1 ;
                0 ;
                0 ;
                0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
            
        end
          
         % Test for construction
         %TODO: 
        function testConstructorWithDec11Ch4Ord00Ang0Pi4(testCase)
            
            % Parameters
            decch = [ 1 1 4 ];
            ord = [ 0 0 ];
            ang = [ 0 0 0 0 0 pi/2 ];
            
            % Expected values
            import saivdr.dictionary.utility.*            
            omgs = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            matrixW0 = step(omgs,ang(1),1);
            matrixU0 = step(omgs,ang(2),1);
            coefExpctd(:,:,1,1) = ...
                blkdiag(matrixW0, matrixU0) * ...
                [ 1 0 0 0 ].';
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
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
            dec = [ 4 4 ];
            ord = [ 0 0 ];
            sizeInvalid = [2 2];
            ang = 2*pi*rand(sizeInvalid);
            
            % Expected value
            sizeExpctd = 120;
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('Length of angles must be %d',...
                sizeExpctd);
            
            % Instantiation of target class
            try
                import saivdr.dictionary.cnsoltx.*
                CplxOvsdLpPuFb2dTypeIVm0System(...
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
            decch = [ 4 4 5 ];
            ord = [ 0 0 ];
            
            % Expected values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = '#Channels must be even.';
            
            % Instantiation of target class
            try
                import saivdr.dictionary.cnsoltx.*
                CplxOvsdLpPuFb2dTypeIVm0System(...
                    'DecimationFactor',decch(1:2),...
                    'NumberOfChannels',decch(3:end),...
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
            dec = [ 2 2 ];
            ord = [ 0 0 ];
            sym = [ 0 0 0 0 ];
            ang = [ 0 0 0 0 0 0 ];
            mus = [ 1  1  1  1 ];

            % Expected values
            coefExpctd = 1/2*[...
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym,...
                'Angles',ang,...
                'Mus',mus);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 2
        function testConstructorWithOrd22(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,2,1) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,3,1) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,1,2) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,2,2) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            coefExpctd(:,:,3,2) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,1,3) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,2,3) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            coefExpctd(:,:,3,3) = [
                0     0     0     0
                0     0     0     0
                0     0     0     0
                0     0     0     0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            %coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            %MEMO: RelTol????AbsTol?????X
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for construction with order 4 4
        function testConstructorWithOrd44(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 4 4 ];
            nDecs = prod(dec);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
            for idx = 1:nDecs
                coefPhaseShift(idx,:,:) = coefActual(idx,:,:);
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

        %{
        % Test for construction with order 5 5
        function testConstructorWithOrd55(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 5 5 ];
            ang = 2*pi*rand(1,12);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
                         coefPhaseShift = zeros(size(coefActual));             %TODO:????????             for idx = 1:nCh                 coefPhaseShift(idx,:,:) = exp(-1i*sym(idx))*coefActual(idx,:,:);             end             coefDiff = coefPhaseShift(:,:)-fliplr(conj(coefPhaseShift(:,:)));
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
        %}
        
        % Test dec 4 4 order 2 2
        function testConstructorWithDec44Ord22(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                ];
            
            coefExpctd(:,:,2,1) = [
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                ];
            
            coefExpctd(:,:,3,1) = [
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                ];
            
            coefExpctd(:,:,1,2) = [
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                ];
            
            coefExpctd(:,1:4,2,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.000000000000007 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i];
            
            coefExpctd(:,5:8,2,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i];
            
            coefExpctd(:,9:12,2,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i];
            
            coefExpctd(:,13:16,2,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i];
            
            
            coefExpctd(:,:,3,2) = [
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                ];
            
            coefExpctd(:,:,1,3) = [
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                ];
            
            coefExpctd(:,:,2,3) = [
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                ];
            
            coefExpctd(:,:,3,3) = [
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
                            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            %coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 4 order 2 2
        function testConstructorWithDec44Ord22Ang(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 2 2 ];
            nDecs = prod(dec);
            nCh = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
            for idx = 1:nDecs
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


        % Test: dec 4 4 order 4 4
        function testConstructorWithDec44Ord44Ang(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 4 4 ];
            nCh = prod(dec);
            nDecs = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            nDecs = prod(dec);
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            for idx = 1:nDecs
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
        
        % Test for angle setting
        function testSetAngles(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 0 0 ];
            angPre = [ pi/2 pi/2 pi/2 pi/2 pi/2 pi/2];
            angPst = [ 0 0 0 0 0 0 ];

            % Expected values
            coefExpctd = 1/2*[...
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
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
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end

        % Test for angle setting
        function testSetMus(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 0 0 ];
            ang = 0;
            musPre = [ 1  1  -1  -1 ];
            musPst = 1;

            % Expected values
            coefExpctd = 1/2*[...
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
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
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for subsref
        function testAnalysisFilterAt(testCase)
            
            % Expected value
            anFiltExpctd1 = 1/2*[  1   1  ;  1   1  ];
            anFiltExpctd2 = 1/2*[  1i  1i ; -1i -1i ];
            anFiltExpctd3 = 1/2*[  1i -1i ;  1i -1i ];
            anFiltExpctd4 = 1/2*[ -1   1  ;  1  -1  ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'OutputMode','AnalysisFilterAt');
            
            % Actual values
            anFiltActual1 = step(testCase.lppufb,[],[],1);
            anFiltActual2 = step(testCase.lppufb,[],[],2);
            anFiltActual3 = step(testCase.lppufb,[],[],3);
            anFiltActual4 = step(testCase.lppufb,[],[],4);
            
            % Evaluation
            import matlab.unittest.constraints.IsLessThan;
            dist = norm(anFiltExpctd1(:)-anFiltActual1(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd2(:)-anFiltActual2(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd3(:)-anFiltActual3(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            dist = norm(anFiltExpctd4(:)-anFiltActual4(:))/2;
            testCase.verifyThat(dist,IsLessThan(1e-14),sprintf('%g',dist));
            
            
        end
        
         function testAnalysisFilters(testCase)
            
            % Expected value
            anFiltExpctd1 = 1/2*[  1   1  ;  1   1  ];
            anFiltExpctd2 = 1/2*[  1i  1i ; -1i -1i ];
            anFiltExpctd3 = 1/2*[  1i -1i ;  1i -1i ];
            anFiltExpctd4 = 1/2*[ -1   1  ;  1  -1  ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'OutputMode','AnalysisFilters');
            
            % Actual values
            anFiltsActual = step(testCase.lppufb,[],[]);
            anFiltActual1 = anFiltsActual(:,:,1);
            anFiltActual2 = anFiltsActual(:,:,2);
            anFiltActual3 = anFiltsActual(:,:,3);
            anFiltActual4 = anFiltsActual(:,:,4);
            
            % Evaluation
            dist = max(abs(anFiltExpctd1(:)-anFiltActual1(:))./abs(anFiltExpctd1(:)));
            testCase.verifyEqual(anFiltActual1,anFiltExpctd1,'AbsTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd2(:)-anFiltActual2(:))./abs(anFiltExpctd2(:)));
            testCase.verifyEqual(anFiltActual2,anFiltExpctd2,'AbsTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd3(:)-anFiltActual3(:))./abs(anFiltExpctd3(:)));
            testCase.verifyEqual(anFiltActual3,anFiltExpctd3,'AbsTol',1e-15,sprintf('%g',dist));
            dist = max(abs(anFiltExpctd4(:)-anFiltActual4(:))./abs(anFiltExpctd4(:)));
            testCase.verifyEqual(anFiltActual4,anFiltExpctd4,'AbsTol',1e-15,sprintf('%g',dist));
            
        end       
        
        % Test dec 2 2 order 0 2
        function testConstructorWithDec22Ord02(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 0 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            %coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 0 2
        function testConstructorWithDec11Ch4Ord02(testCase)
            
            % Parameters
            decch = [ 1 1 4 ];
            ord = [ 0 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  ;
                0  ;
                0  ;
                0  ];
            
            coefExpctd(:,:,1,2) = [
                1 ;
                0 ;
                0 ;
                0 ];
            
            coefExpctd(:,:,1,3) = [
                0  ;
                0  ;
                0  ;
                0  ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end     
        
        % Test dec 1 1 ch 4 order 2 2
        function testConstructorWithDec11Ch4Ord22(testCase)
            
            % Parameters
            decch = [ 1 1 4 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [...
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2,1) =  [...
                0;
                0;
                0;
                0];

            coefExpctd(:,:,3,1) = [...
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,2,1) =  [...
                0;
                0;
                0;
                0];

            coefExpctd(:,:,2,2) = [...
                1;
                0;
                0;
                0];
            
            coefExpctd(:,:,2,3) =  [...
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3,1) = [...
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3,2) =  [...
                0;
                0;
                0;
                0];
            
            coefExpctd(:,:,3,3) = [...
                0;
                0;
                0;
                0];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
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
        function testConstructorWithDec22Ch4Ord02(testCase)
            
            % Parameters
            decch = [ 2 2 4 ];
            ord = [ 0 2 ];
            ang = 0;
            
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            %coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
         % Test dec 2 2 order 2 0
        function testConstructorWithDec22Ch4Ord20(testCase)
            
            % Parameters
            decch = [ 2 2 4 ];
            ord = [ 2 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            %coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        
        
        % Test for construction with order 0 2
        function testConstructorWithDec22Ord02Ang(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 0 2 ];
            nDecs = prod(dec);
            nCh = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nDecs
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
         % Test dec 2 2 order 2 0
        function testConstructorWithDec22Ord20(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 2 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2) = 1/2*[...
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            coefExpctd(:,:,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            %coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 2 0
        function testConstructorWithDec11Ch4Ord20(testCase)
            
            % Parameters
            decch = [ 1 1 4 ];
            ord = [ 2 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0  ;
                0  ;
                0  ;
                0  ];
            
            coefExpctd(:,:,2) = [
                1 ;
                0 ;
                0 ;
                0 ];
            
            coefExpctd(:,:,3) = [
                0  ;
                0  ;
                0  ;
                0  ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 2 0
        function testConstructorWithDec22Ord20Ang(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 2 0 ];
            nDecs = prod(dec);
            nCh = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nDecs
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
        
        % Test dec 2 2 order 2 2
        function testConstructorWithDec22Ord22(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 2 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            %coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 2 2
        function testConstructorWithDec22Ord22Ang(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 2 2 ];
            nDecs = prod(dec);
            nCh = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
            for idx = 1:nDecs
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
        
        % Test dec 1 1 ch 4 order 4 4
        function testConstructorWithDec11Ch4Ord44(testCase)
            
            % Parameters
            decch = [ 1 1 4 ];
            ord = [ 4 4 ];
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
              % Expected values
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
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
        
        
         % Test dec 2 2 ch 4 order 2 2
        function testConstructorWithDec22Ch4Ord22Ang(testCase)
            
            % Parameters
            decch = [ 2 2 4 ];
            ord = [ 2 2 ];
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
              % Expected values
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
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
        
        % Test dec 2 2 order 0 4
        function testConstructorWithDec22Ord04(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 0 4 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            coefExpctd(:,:,1,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            %coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 0 4
        function testConstructorWithDec22Ord04Ang(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 0 4 ];
            nDecs = prod(dec);
            nCh = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
            for idx = 1:nDecs
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
        
        % Test dec 2 2 order 0 4
        function testConstructorWithDec22Ord40(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 4 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            coefExpctd(:,:,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            %coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test for construction with order 4 0
        function testConstructorWithDec22Ord40Ang(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 4 0 ];
            nDecs = prod(dec);
            nCh = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
            for idx = 1:nDecs
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
        
        % Test dec 2 2 order 2 2
        function testConstructorWithDec22Ord44(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 4 4 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,2) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,3) = 1/2*[
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            coefExpctd(:,:,4,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,3) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,1) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,4) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,1,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,2,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,3,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,4,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            coefExpctd(:,:,5,5) = [
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ;
                0  0  0  0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            %coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 2 2
        function testConstructorWithDec22Ord44Ang(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 4 4 ];
            nDecs = prod(dec);
            nCh = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
            for idx = 1:nDecs
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
        
        % Test dec 4 4 order 0 2
        function testConstructorWithDec44Ord02(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 0 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = zeros(16,16);
            
            coefExpctd(:,1:4,1,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.000000000000007 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i];
            
            coefExpctd(:,5:8,1,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i];
            
            coefExpctd(:,9:12,1,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i];
            
            coefExpctd(:,13:16,1,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i];
            
            coefExpctd(:,:,1,3) = zeros(16,16);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 0 2
        function testConstructorWithDec44Ord02Ang(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 0 2 ];
            nDecs = prod(dec);
            nCh = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
            for idx = 1:nDecs
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
        
        % Test dec 4 4 order 2 0
        function testConstructorWithDec44Ord20(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 2 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = zeros(16,16);
            
            coefExpctd(:,1:4,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.000000000000007 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i];
            
            coefExpctd(:,5:8,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i];
            
            coefExpctd(:,9:12,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i];
            
            coefExpctd(:,13:16,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i];
            
            coefExpctd(:,:,3) = zeros(16,16);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 4 order 2 0
        function testConstructorWithDec44Ord20Ang(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 2 0 ];
            nDecs = prod(dec);
            nCh = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
            for idx = 1:nDecs
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
        
        % Test dec 4 4 order 0 4
        function testConstructorWithDec44Ord04(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 0 4 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = zeros(16,16);
            
            coefExpctd(:,:,1,2) = zeros(16,16);
            
            coefExpctd(:,1:4,1,3) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.000000000000007 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i];
            
            coefExpctd(:,5:8,1,3) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i];
            
            coefExpctd(:,9:12,1,3) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i];
            
            coefExpctd(:,13:16,1,3) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i];
            
            coefExpctd(:,:,1,4) = zeros(16,16);
            
            coefExpctd(:,:,1,5) = zeros(16,16);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 4 order 0 4
        function testConstructorWithDec44Ord04Ang(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 0 4 ];
            nDecs = prod(dec);
            nCh = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
            for idx = 1:nDecs
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
        
        % Test dec 4 4 order 4 0
        function testConstructorWithDec44Ord40(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 4 0 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1) = zeros(16,16);
            
            coefExpctd(:,:,2) = zeros(16,16);
            
            coefExpctd(:,1:4,3) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.000000000000007 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i];
            
            coefExpctd(:,5:8,3) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i];
            
            coefExpctd(:,9:12,3) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i];
            
            coefExpctd(:,13:16,3) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i];
            
            coefExpctd(:,:,4) = zeros(16,16);
            
            coefExpctd(:,:,5) = zeros(16,16);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 4 order 4 0
        function testConstructorWithDec44Ord40Ang(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 4 0 ];
            nDecs = prod(dec);
            nCh = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
            for idx = 1:nDecs
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
        
        % Test dec 2 2 order 2 4
        function testConstructorWithDec22Ord24(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 2 4 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = zeros(4,4);
            
            coefExpctd(:,:,2,1) = zeros(4,4);
            
            coefExpctd(:,:,3,1) = zeros(4,4);
            
            coefExpctd(:,:,1,2) = zeros(4,4);
            
            coefExpctd(:,:,2,2) = zeros(4,4);
            
            coefExpctd(:,:,3,2) = zeros(4,4);
            
            coefExpctd(:,:,1,3) = zeros(4,4);
            
            coefExpctd(:,:,2,3) = 1/2*[...
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            coefExpctd(:,:,3,3) = zeros(4,4);
            
            coefExpctd(:,:,1,4) = zeros(4,4);
            
            coefExpctd(:,:,2,4) = zeros(4,4);
            
            coefExpctd(:,:,3,4) = zeros(4,4);
            
            coefExpctd(:,:,1,5) = zeros(4,4);
            
            coefExpctd(:,:,2,5) = zeros(4,4);
            
            coefExpctd(:,:,3,5) = zeros(4,4);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                       
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 2 4
        function testConstructorWithDec22Ord24Ang(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 2 4 ];
            nCh = prod(dec);
            nDecs = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym,...
                'Angles',ang);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
            for idx = 1:nDecs
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
        
        % Test dec 4 4 order 2 4
        function testConstructorWithDec44Ord24(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 2 4 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = zeros(16,16);
            
            coefExpctd(:,:,2,1) = zeros(16,16);
            
            coefExpctd(:,:,3,1) = zeros(16,16);
            
            coefExpctd(:,:,1,2) = zeros(16,16);
            
            coefExpctd(:,:,2,2) = zeros(16,16);
            
            coefExpctd(:,:,3,2) = zeros(16,16);
            
            coefExpctd(:,:,1,3) = zeros(16,16);
            
            coefExpctd(:,1:4,2,3) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.000000000000007 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i];
            
            coefExpctd(:,5:8,2,3) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i];
            
            coefExpctd(:,9:12,2,3) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i];
            
            coefExpctd(:,13:16,2,3) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i];
            
            coefExpctd(:,:,3,3) = zeros(16,16);
            
            coefExpctd(:,:,1,4) = zeros(16,16);
            
            coefExpctd(:,:,2,4) = zeros(16,16);
            
            coefExpctd(:,:,3,4) = zeros(16,16);
            
            coefExpctd(:,:,1,5) = zeros(16,16);
            
            coefExpctd(:,:,2,5) = zeros(16,16);
            
            coefExpctd(:,:,3,5) = zeros(16,16);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 4 order 2 4
        function testConstructorWithDec44Ord24Ang(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 2 4 ];
            nCh = prod(dec);
            nDecs = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:????????
            for idx = 1:nDecs
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
        
        % Test dec 2 2 order 4 2
        function testConstructorWithDec22Ord42(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 4 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = zeros(4,4);
            
            coefExpctd(:,:,2,1) = zeros(4,4);
            
            coefExpctd(:,:,3,1) = zeros(4,4);
            
            coefExpctd(:,:,4,1) = zeros(4,4);
            
            coefExpctd(:,:,5,1) = zeros(4,4);
            
            coefExpctd(:,:,1,2) = zeros(4,4);
            
            coefExpctd(:,:,2,2) = zeros(4,4);
            
            coefExpctd(:,:,3,2) = 1/2*[...
                 1 ,  1 ,  1 ,  1 ;
                 1i, -1i,  1i, -1i;
                 1i,  1i, -1i, -1i;
                -1 ,  1 ,  1 , -1 ];
            
            coefExpctd(:,:,4,2) = zeros(4,4);
            
            coefExpctd(:,:,5,2) = zeros(4,4);
            
            coefExpctd(:,:,1,3) = zeros(4,4);
            
            coefExpctd(:,:,2,3) = zeros(4,4);
            
            coefExpctd(:,:,3,3) = zeros(4,4);
            
            coefExpctd(:,:,4,3) = zeros(4,4);
            
            coefExpctd(:,:,5,3) = zeros(4,4);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 2 2 order 4 2
        function testConstructorWithDec22Ord42Ang(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 4 2 ];
            nCh = prod(dec);
            nDecs = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nDecs
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
        
        % Test dec 4 4 order 4 2
        function testConstructorWithDec44Ord42(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 4 2 ];
            ang = 0;
            
            % Expected values
            coefExpctd(:,:,1,1) = zeros(16,16);
            
            coefExpctd(:,:,2,1) = zeros(16,16);
            
            coefExpctd(:,:,3,1) = zeros(16,16);
            
            coefExpctd(:,:,4,1) = zeros(16,16);
            
            coefExpctd(:,:,5,1) = zeros(16,16);
            
            coefExpctd(:,:,1,2) = zeros(16,16);
            
            coefExpctd(:,:,2,2) = zeros(16,16);
            
            coefExpctd(:,1:4,3,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.000000000000007 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i];
            
            coefExpctd(:,5:8,3,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i];
            
            coefExpctd(:,9:12,3,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 + 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i;
                -0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i];
            
            coefExpctd(:,13:16,3,2) = [
                0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i,0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 + 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                -0.176776695296637 + 0.176776695296637i,0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i,0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 + 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i,0.00000000000000 - 0.250000000000000i;
                0.176776695296637 - 0.176776695296637i,0.176776695296637 + 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i,0.250000000000000 - 0.00000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                -0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i,-0.176776695296637 - 0.176776695296637i;
                0.00000000000000 - 0.250000000000000i,0.250000000000000 - 0.00000000000000i,0.00000000000000 + 0.250000000000000i,-0.250000000000000 + 0.00000000000000i;
                0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i,0.176776695296637 - 0.176776695296637i,-0.176776695296637 + 0.176776695296637i;
                0.250000000000000 - 0.00000000000000i,0.00000000000000 - 0.250000000000000i,-0.250000000000000 + 0.00000000000000i,0.00000000000000 + 0.250000000000000i];
            
            coefExpctd(:,:,4,2) = zeros(16,16);
            
            coefExpctd(:,:,5,2) = zeros(16,16);
            
            coefExpctd(:,:,1,3) = zeros(16,16);
            
            coefExpctd(:,:,2,3) = zeros(16,16);
            
            coefExpctd(:,:,3,3) = zeros(16,16);
            
            coefExpctd(:,:,4,3) = zeros(16,16);
            
            coefExpctd(:,:,5,3) = zeros(16,16);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        
        % Test dec 4 4 order 4 2
        function testConstructorWithDec44Ord42Ang(testCase)
            
            % Parameters
            dec = [ 4 4 ];
            ord = [ 4 2 ];
            nCh = prod(dec);
            nDecs = prod(dec);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nDecs nDecs ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
            % Actual values
            coefActual = step(testCase.lppufb,ang,[]);
            
            % Evaluation
            testCase.verifySize(coefActual,dimExpctd);
            
            % Check symmetry
            import matlab.unittest.constraints.IsLessThan;
            coefPhaseShift = zeros(size(coefActual));
            %TODO:
            for idx = 1:nDecs
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
        
        % Test dec 2 2 ch 4 order 4 4
        function testConstructorWithDec22Ch4Ord44Ang(testCase)
            
            % Parameters
            decch = [ 2 2 4 ];
            ord = [ 4 4 ];
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
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

        % Test dec 2 2 ch 6 order 2 2
        function testConstructorWithDec22Ch6Ord22Ang(testCase)
            
            % Parameters
            decch = [ 2 2 6 ];
            ord = [ 2 2 ];
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
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

        % Test dec 2 2 ch 6 order 4 4
        function testConstructorWithDec22Ch6Ord44Ang(testCase)
            
            % Parameters
            decch = [ 2 2 6 ];
            ord = [ 4 4 ];
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
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

        % Test dec 2 2 ch 8 order 2 2
        function testConstructorWithDec22Ch8Ord22Ang(testCase)
            
            % Parameters
            decch = [ 2 2 8 ];
            ord = [ 2 2 ];
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
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
        
        % Test dec 2 2 ch 8 order 4 4
        function testConstructorWithDec22Ch8Ord44Ang(testCase)
            
            % Parameters
            decch = [ 2 2 8 ];
            ord = [ 4 4 ];
            nCh = decch(3);
            nDec = decch(1)*decch(2);
            
            ang = randn(nCh*(nCh-1)/2+sum(ord)*(nCh*(nCh-2)/4+floor(nCh/4)),1);
            sym = randn(1,nCh);
            
            % Expected values
            dimExpctd = [nCh nDec ord(1)+1 ord(2)+1];
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*                        
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',ord,...
                'Symmetry',sym);
            
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
 
        
        % Test for ParameterMatrixSet
        function testParameterMatrixSet(testCase)
            
            % Preparation
            mstab = [ 4 4 ];
            
            % Expected value
            import saivdr.dictionary.utility.ParameterMatrixContainer
            paramExpctd = ParameterMatrixContainer(...
                'MatrixSizeTable',mstab);
            step(paramExpctd,eye(4),1);
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'OutputMode','ParameterMatrixSet');
            
            % Actual values
            paramActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
            testCase.verifyEqual(paramExpctd, paramActual);
            
        end
        
        %{
        % Test for construction
        function testConstructorWithOrd10(testCase)
            
            % Parameters
            dec = [ 2 2 ];
            ord = [ 1 0 ];
            
            % Expected values
            coefExpctd(:,:,1,1) = 1/2 * [
                0  1  0  1 ;
                0  1  0 -1 ;
                0 -1  0 -1 ;
                0 -1  0  1 ];
            
            coefExpctd(:,:,2,1) = 1/2 * [
                1  0  1  0 ;
               -1  0  1  0 ;
                1  0  1  0;
               -1  0  1  0];            
            
            % Instantiation of target class
            import saivdr.dictionary.cnsoltx.*            
            testCase.lppufb = CplxOvsdLpPuFb2dTypeIVm0System(...
                'DecimationFactor',dec,...
                'PolyPhaseOrder',ord);
            
            % Actual values
            coefActual = step(testCase.lppufb,[],[]);
            
            % Evaluation
           coefDist = max(abs(coefExpctd(:)-coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-14,sprintf('%g',coefDist));
            
        end
        %}

    end
    
end
