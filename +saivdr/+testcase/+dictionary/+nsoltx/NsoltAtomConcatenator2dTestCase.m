classdef NsoltAtomConcatenator2dTestCase < matlab.unittest.TestCase
    %MODULEBLOCKDCT2DTESTCASE Test case for ModuleBlockDct2d
    %
    % SVN identifier:
    % $Id: NsoltAtomConcatenator2dTestCase.m 683 2015-05-29 08:22:13Z sho $
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
            nchExpctd = [ 2 2 ];
            fpeExpctd = false;
            typExpctd = 'Type I';
            ordExpctd = [ 0 0 ];
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d();
            
            % Actual values
            fpeActual = get(testCase.module,'IsPeriodicExt');
            nchActual = [ 
                get(testCase.module,'NumberOfSymmetricChannels') ...
                get(testCase.module,'NumberOfAntisymmetricChannels') ];
            typActual = get(testCase.module,'NsoltType');
            ordActual = get(testCase.module,'PolyPhaseOrder');
            
            % Evaluation
            testCase.verifyEqual(fpeActual,fpeExpctd);
            testCase.verifyEqual(nchActual,nchExpctd);
            testCase.verifyEqual(typActual,typExpctd);
            testCase.verifyEqual(ordActual,ordExpctd);
            
        end
        
        function testConstractionTypeII(testCase)
            
            % Expected values
            nchExpctd = [ 3 2 ];
            fpeExpctd = false;
            typExpctd = 'Type II';
            ordExpctd = [ 0 0 ];
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nchExpctd(1),...
                'NumberOfAntisymmetricChannels',nchExpctd(2));
            
            % Actual values
            fpeActual = get(testCase.module,'IsPeriodicExt');
            nchActual = [
                get(testCase.module,'NumberOfSymmetricChannels') ...
                get(testCase.module,'NumberOfAntisymmetricChannels') ];
            typActual = get(testCase.module,'NsoltType');
            ordActual = get(testCase.module,'PolyPhaseOrder');
            
            % Evaluation
            testCase.verifyEqual(fpeActual,fpeExpctd);
            testCase.verifyEqual(nchActual,nchExpctd);
            testCase.verifyEqual(typActual,typExpctd);
            testCase.verifyEqual(ordActual,ordExpctd);
            
        end
        
        function testStepTypeI(testCase)

            % Parameters
            height = 16;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 0 0 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            pmCoefs = [I0(:) ; I0(:)];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;            
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end           
        
        function testStepTypeII(testCase)

            % Parameters
            height = 16;
            width  = 16;
            nch   = [ 3 2 ];
            ord   = [ 0 0 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            pmCoefs = [I0(:) ; I0(:)];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end                   
        
        function testStepOrd22Ch22(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 2 ];
            nch   = [ 2 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(nch(2));
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-9);
            
        end         
        
        %TODO: テストケースの名称を変更する
        function testStepOrd22Ch22U0(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 2 ];
            nch   = [ 2 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            V0 = dctmtx(sum(nch));
            In = eye(nch(2));
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; V0(:) ;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            cfsExpctd = V0.'*cfsExpctd;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end                 
        
        function testStepOrd20Ch22(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 0 ];
            nch   = [ 2 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(nch(2));
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end          
              
        function testStepOrd02Ch22(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 0 2 ];
            nch   = [ 2 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(nch(2));
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end          
     
        function testStepOrd44Ch22(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 4 4 ];
            nch   = [ 2 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(nch(2));
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end         
        
        function testStepOrd42Ch22(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 4 2 ];
            nch   = [ 2 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(nch(2));
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end           
    
        function testStepOrd24Ch22(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 4 ];
            nch   = [ 2 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(nch(2));
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end               
        
        function testStepOrd22Ch44(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 2 ];
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            In = eye(nch(2));
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                In(:) ; -In(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end                       
        
        function testStepOrd22Ch32(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 2 ];
            nch   = [ 3 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(nch(1));
            In = eye(nch(2));
            Ux = blkdiag(-In,1);
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end  
        
        %TODO: テストケース名を変更する
        function testStepOrd22Ch32U0(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 2 ];
            nch   = [ 3 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            V0 = dctmtx(sum(nch));
            Ix = eye(nch(1));
            In = eye(nch(2));
            Ux = blkdiag(-In,1);
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; V0(:) ;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            cfsExpctd = V0.'*cfsExpctd;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end      
        
        function testStepOrd20Ch32(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 0 ];
            nch   = [ 3 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(nch(1));
            In = eye(nch(2));
            Ux = blkdiag(-In,1);
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end         
        
        function testStepOrd02Ch32(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 0 2 ];
            nch   = [ 3 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(nch(1));
            In = eye(nch(2));
            Ux = blkdiag(-In,1);
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end                 
        
        function testStepOrd44Ch32(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 4 4 ];
            nch   = [ 3 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(nch(1));
            In = eye(nch(2));
            Ux = blkdiag(-In,1);
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end                 
        
        function testStepOrd42Ch32(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 4 2 ];
            nch   = [ 3 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(nch(1));
            In = eye(nch(2));
            Ux = blkdiag(-In,1);
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end                         
        
        function testStepOrd24Ch32(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 4 ];
            nch   = [ 3 2 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(nch(1));
            In = eye(nch(2));
            Ux = blkdiag(-In,1);
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
            
        end                                 
        
        function testStepOrd22Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            ord   = [ 2 2 ];
            nch   = [ 5 4 ];
            coefs = randn(sum(nch), height*width)+1i*randn(sum(nch), height*width);
            scale = [ height width ];
            I0 = eye(sum(nch));
            Ix = eye(nch(1));
            In = eye(nch(2));
            Ux = blkdiag(-In,1);
            angsB = pi/2*ones(floor(nch(2)/2),1);
            pmCoefs = [ I0(:) ; I0(:) ;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB;
                In(:) ; -In(:) ; angsB;
                Ix(:) ;  Ux(:) ; angsB; ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
            testCase.module = NsoltAtomConcatenator2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-9);
            
        end
        
%         function testStepOrd22Ch23(testCase)
% 
%             % Parameters
%             height = 16;
%             width  = 16;
%             ord   = [ 2 2 ];
%             nch   = [ 2 3 ];
%             coefs = randn(sum(nch), height*width);
%             scale = [ height width ];
%             In = eye(nch(1));
%             Ix = eye(nch(2));
%             pmCoefs = [ In(:) ; Ix(:) ; -In(:) ;  Ix(:) ; -In(:) ; Ix(:) ];
%             
%             % Expected values
%             ordExpctd = ord;
%             cfsExpctd = coefs;
%                         
%             % Instantiation
%             import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
%             testCase.module = NsoltAtomConcatenator2d(...
%                 'NumberOfSymmetricChannels',nch(1),...
%                 'NumberOfAntisymmetricChannels',nch(2));
%             set(testCase.module,'PolyPhaseOrder',ord);
%             
%             % Actual values
%             ordActual = get(testCase.module,'PolyPhaseOrder');
%             cfsActual = step(testCase.module,coefs,scale,pmCoefs);
%             
%             % Evaluation
%             testCase.verifyEqual(ordActual,ordExpctd);
%             testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
%             
%         end       
% 
%         function testStepOrd22Ch23W0(testCase)
% 
%             % Parameters
%             height = 16;
%             width  = 16;
%             ord   = [ 2 2 ];
%             nch   = [ 2 3 ];
%             coefs = randn(sum(nch), height*width);
%             scale = [ height width ];
%             In = eye(nch(1));
%             Ix = eye(nch(2));
%             W0 = dctmtx(nch(1));
%             pmCoefs = [ 
%                 W0(:) ;
%                 Ix(:) ; 
%                 -In(:) ; 
%                 Ix(:) ;                
%                 -In(:) ; 
%                 Ix(:) ];
%             
%             % Expected values
%             ordExpctd = ord;
%             cfsExpctd = coefs;
%             cfsExpctd(1:nch(1),:) = W0.'*cfsExpctd(1:nch(1),:);
%                         
%             % Instantiation
%             import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
%             testCase.module = NsoltAtomConcatenator2d(...
%                 'NumberOfSymmetricChannels',nch(1),...
%                 'NumberOfAntisymmetricChannels',nch(2));
%             set(testCase.module,'PolyPhaseOrder',ord);
%             
%             % Actual values
%             ordActual = get(testCase.module,'PolyPhaseOrder');
%             cfsActual = step(testCase.module,coefs,scale,pmCoefs);
%             
%             % Evaluation
%             testCase.verifyEqual(ordActual,ordExpctd);
%             testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
%             
%         end      
% 
%         function testStepOrd20Ch23(testCase)
% 
%             % Parameters
%             height = 16;
%             width  = 16;
%             ord   = [ 2 0 ];
%             nch   = [ 2 3 ];
%             coefs = randn(sum(nch), height*width);
%             scale = [ height width ];
%             In = eye(nch(1));
%             Ix = eye(nch(2));
%             pmCoefs = [ 
%                 -In(:) ; 
%                 Ix(:) ;
%                 -In(:) ; 
%                 Ix(:) ];
%             
%             % Expected values
%             ordExpctd = ord;
%             cfsExpctd = coefs;
%                         
%             % Instantiation
%             import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
%             testCase.module = NsoltAtomConcatenator2d(...
%                 'NumberOfSymmetricChannels',nch(1),...
%                 'NumberOfAntisymmetricChannels',nch(2));
%             set(testCase.module,'PolyPhaseOrder',ord);
%             
%             % Actual values
%             ordActual = get(testCase.module,'PolyPhaseOrder');
%             cfsActual = step(testCase.module,coefs,scale,pmCoefs);
%             
%             % Evaluation
%             testCase.verifyEqual(ordActual,ordExpctd);
%             testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
%             
%         end         
% 
%         function testStepOrd02Ch23(testCase)
% 
%             % Parameters
%             height = 16;
%             width  = 16;
%             ord   = [ 0 2 ];
%             nch   = [ 2 3 ];
%             coefs = randn(sum(nch), height*width);
%             scale = [ height width ];
%             In = eye(nch(1));
%             Ix = eye(nch(2));
%             pmCoefs = [ 
%                 -In(:) ; 
%                 Ix(:) ;
%                 -In(:) ; 
%                 Ix(:) ];
%             
%             % Expected values
%             ordExpctd = ord;
%             cfsExpctd = coefs;
%                         
%             % Instantiation
%             import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
%             testCase.module = NsoltAtomConcatenator2d(...
%                 'NumberOfSymmetricChannels',nch(1),...
%                 'NumberOfAntisymmetricChannels',nch(2));
%             set(testCase.module,'PolyPhaseOrder',ord);
%             
%             % Actual values
%             ordActual = get(testCase.module,'PolyPhaseOrder');
%             cfsActual = step(testCase.module,coefs,scale,pmCoefs);
%             
%             % Evaluation
%             testCase.verifyEqual(ordActual,ordExpctd);
%             testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
%             
%         end                 
%         
%         function testStepOrd44Ch23(testCase)
% 
%             % Parameters
%             height = 16;
%             width  = 16;
%             ord   = [ 4 4 ];
%             nch   = [ 2 3 ];
%             coefs = randn(sum(nch), height*width);
%             scale = [ height width ];
%             In = eye(nch(1));
%             Ix = eye(nch(2));
%             pmCoefs = [ 
%                 In(:) ; 
%                 Ix(:) ;
%                 -In(:) ; 
%                 Ix(:) ;
%                 -In(:) ; 
%                 Ix(:) ;
%                 -In(:) ; 
%                 Ix(:) ;                
%                 -In(:) ; 
%                 Ix(:) ];
%             
%             % Expected values
%             ordExpctd = ord;
%             cfsExpctd = coefs;
%                         
%             % Instantiation
%             import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
%             testCase.module = NsoltAtomConcatenator2d(...
%                 'NumberOfSymmetricChannels',nch(1),...
%                 'NumberOfAntisymmetricChannels',nch(2));
%             set(testCase.module,'PolyPhaseOrder',ord);
%             
%             % Actual values
%             ordActual = get(testCase.module,'PolyPhaseOrder');
%             cfsActual = step(testCase.module,coefs,scale,pmCoefs);
%             
%             % Evaluation
%             testCase.verifyEqual(ordActual,ordExpctd);
%             testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
%             
%         end                 
% 
%         function testStepOrd42Ch23(testCase)
% 
%             % Parameters
%             height = 16;
%             width  = 16;
%             ord   = [ 4 2 ];
%             nch   = [ 2 3 ];
%             coefs = randn(sum(nch), height*width);
%             scale = [ height width ];
%             In = eye(nch(1));
%             Ix = eye(nch(2));
%             pmCoefs = [ 
%                 -In(:) ; 
%                 Ix(:) ;
%                 -In(:) ; 
%                 Ix(:) ;
%                 -In(:) ; 
%                 Ix(:) ;                
%                 -In(:) ; 
%                 Ix(:) ];
%             
%             % Expected values
%             ordExpctd = ord;
%             cfsExpctd = coefs;
%                         
%             % Instantiation
%             import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
%             testCase.module = NsoltAtomConcatenator2d(...
%                 'NumberOfSymmetricChannels',nch(1),...
%                 'NumberOfAntisymmetricChannels',nch(2));
%             set(testCase.module,'PolyPhaseOrder',ord);
%             
%             % Actual values
%             ordActual = get(testCase.module,'PolyPhaseOrder');
%             cfsActual = step(testCase.module,coefs,scale,pmCoefs);
%             
%             % Evaluation
%             testCase.verifyEqual(ordActual,ordExpctd);
%             testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
%             
%         end                         
% 
%         function testStepOrd24Ch23(testCase)
% 
%             % Parameters
%             height = 16;
%             width  = 16;
%             ord   = [ 2 4 ];
%             nch   = [ 2 3 ];
%             coefs = randn(sum(nch), height*width);
%             scale = [ height width ];
%             In = eye(nch(1));
%             Ix = eye(nch(2));
%             pmCoefs = [ 
%                 -In(:) ; 
%                 Ix(:) ;
%                 -In(:) ; 
%                 Ix(:) ;
%                 -In(:) ; 
%                 Ix(:) ;                
%                 -In(:) ; 
%                 Ix(:) ];
%             
%             % Expected values
%             ordExpctd = ord;
%             cfsExpctd = coefs;
%                         
%             % Instantiation
%             import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
%             testCase.module = NsoltAtomConcatenator2d(...
%                 'NumberOfSymmetricChannels',nch(1),...
%                 'NumberOfAntisymmetricChannels',nch(2));
%             set(testCase.module,'PolyPhaseOrder',ord);
%             
%             % Actual values
%             ordActual = get(testCase.module,'PolyPhaseOrder');
%             cfsActual = step(testCase.module,coefs,scale,pmCoefs);
%             
%             % Evaluation
%             testCase.verifyEqual(ordActual,ordExpctd);
%             testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
%             
%         end                                 
% 
%         function testStepOrd22Ch45(testCase)
% 
%             % Parameters
%             height = 16;
%             width  = 16;
%             ord   = [ 2 2 ];
%             nch   = [ 4 5 ];
%             coefs = randn(sum(nch), height*width);
%             scale = [ height width ];
%             In = eye(nch(1));
%             Ix = eye(nch(2));
%             pmCoefs = [ 
%                 In(:) ; 
%                 Ix(:) ;
%                 -In(:) ; 
%                 Ix(:) ;                
%                 -In(:) ; 
%                 Ix(:) ];
%             
%             % Expected values
%             ordExpctd = ord;
%             cfsExpctd = coefs;
%                         
%             % Instantiation
%             import saivdr.dictionary.nsoltx.NsoltAtomConcatenator2d
%             testCase.module = NsoltAtomConcatenator2d(...
%                 'NumberOfSymmetricChannels',nch(1),...
%                 'NumberOfAntisymmetricChannels',nch(2));
%             set(testCase.module,'PolyPhaseOrder',ord);
%             
%             % Actual values
%             ordActual = get(testCase.module,'PolyPhaseOrder');
%             cfsActual = step(testCase.module,coefs,scale,pmCoefs);
%             
%             % Evaluation
%             testCase.verifyEqual(ordActual,ordExpctd);
%             testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-10);
%             
%         end                                                                                    
                
     end
 
end
