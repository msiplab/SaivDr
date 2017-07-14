classdef CnsoltAtomExtender2dTestCase < matlab.unittest.TestCase
    %NSOLTATOMEXTENDER2DTESTCASE Test case for ModuleBlockDct2d
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627    
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
        
        function testDefaultConstraction(testCase)

            % Expected values
            nchExpctd = 4;
            hchExpctd = 2;
            ordExpctd = [ 0 0 ];            
            fpeExpctd = false;
            typExpctd = 'Type I';
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d();
            
            % Actual values
            fpeActual = get(testCase.module,'IsPeriodicExt');
            nchActual = get(testCase.module,'NumberOfChannels');
            hchActual = get(testCase.module,'NumberOfHalfChannels');
            typActual = get(testCase.module,'NsoltType');
            ordActual = get(testCase.module,'PolyPhaseOrder');
            
            % Evaluation
            testCase.verifyEqual(fpeActual,fpeExpctd);
            testCase.verifyEqual(nchActual,nchExpctd);
            testCase.verifyEqual(hchActual,hchExpctd);
            testCase.verifyEqual(typActual,typExpctd);
            testCase.verifyEqual(ordActual,ordExpctd);
            
        end
        
        function testConstractionTypeII(testCase)

            % Expected values
            nchExpctd = 5;
            hchExpctd = 2;
            ordExpctd = [ 0 0 ];
            fpeExpctd = false;
            typExpctd = 'Type II';
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nchExpctd);
            
            % Actual values
            fpeActual = get(testCase.module,'IsPeriodicExt');
            nchActual = get(testCase.module,'NumberOfChannels');
            hchActual = get(testCase.module,'NumberOfHalfChannels');
            typActual = get(testCase.module,'NsoltType');
            ordActual = get(testCase.module,'PolyPhaseOrder');
            
            % Evaluation
            testCase.verifyEqual(fpeActual,fpeExpctd);
            testCase.verifyEqual(nchActual,nchExpctd);
            testCase.verifyEqual(hchActual,hchExpctd);
            testCase.verifyEqual(typActual,typExpctd);
            testCase.verifyEqual(ordActual,ordExpctd);
            
        end        
        
        function testStepTypeI(testCase)

            % Parameters
            height = 16;
            width  = 16;
            nch   = 4;
            ord   = [ 0 0 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            %S = eye(sum(nch));
            I0 = eye(sum(nch));
            pmCoefs = I0(:);
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end           
        
        function testStepTypeII(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 0 0 ];
            nch   = 5;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            pmCoefs = I0(:);
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);            
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end                   
        
        function testStepOrd22Ch22(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 2 ];
            nch   = 4;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(floor(nch/2));
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end                  
        
        function testStepOrd22Ch22U0(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 2 ];
            nch   = 4;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            U0 = dctmtx(sum(nch));
            In = eye(floor(nch/2));
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [  U0(:) ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            cfsExpctd = U0*cfsExpctd;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end                 
        
        function testStepOrd20Ch22(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 0 ];
            nch   = 4;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(floor(nch/2));
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end        
        
        function testStepOrd02Ch22(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 0 2 ];
            nch   = 4;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(floor(nch/2));
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end                
        
        function testStepOrd44Ch22(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 4 4 ];
            nch   = 4;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(floor(nch/2));
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ; ];                        
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end                        
        
        function testStepOrd42Ch22(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 4 2 ];
            nch   = 4;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(floor(nch/2));
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end                                
        
       function testStepOrd24Ch22(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 4 ];
            nch   = 4;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(floor(nch/2));
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
       end                                        
        
       function testStepOrd22Ch44(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 2 ];
            nch   = 8;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(floor(nch/2));
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                In(:) ; -In(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end                                               
        
       function testStepOrd22Ch32(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 2 ];
            nch   = 5;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;            
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);            
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');            
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                                    
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
       end        
       
       function testStepOrd22Ch32V0(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 2 ];
            nch   = 5;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            V0 = dctmtx(sum(nch));
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Ux = blkdiag(-In,1);
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ V0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            cfsExpctd = V0*cfsExpctd;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end                 

        function testStepOrd20Ch32(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 0 ];
            nch   = 5;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Ux = blkdiag(-In,1);
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end        
        
        function testStepOrd02Ch32(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 0 2 ];
            nch   = 5;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Ux = blkdiag(-In,1);
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end                
        
        function testStepOrd44Ch32(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 4 4 ];
            nch   = 5;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Ux = blkdiag(-In,1);
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end                        
        
        function testStepOrd42Ch32(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 4 2 ];
            nch   = 5;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Ux = blkdiag(-In,1);
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end                                
        
       function testStepOrd24Ch32(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 4 ];
            nch   = 5;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Ux = blkdiag(-In,1);
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
       end                                        
        
       function testStepOrd22Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 2 ];
            nch   = 9;
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Ux = blkdiag(-In,1);
            angsB = zeros(floor(floor(nch/2)/2),1);
            pmCoefs = [ I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Ux(:) ; angsB ; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomExtender2d
            testCase.module = CnsoltAtomExtender2d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);                        
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
       end
 
    end
 
end
