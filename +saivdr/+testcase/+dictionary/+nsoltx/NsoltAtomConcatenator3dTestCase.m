classdef NsoltAtomConcatenator3dTestCase < matlab.unittest.TestCase
    %NSOLTATOMCONCATENATOR3DTESTCASE Test case for ModuleBlockDct3d
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2017, Shogo MURAMATSU
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
        
        function testDefaultConstraction(testCase)

            % Expected values
            nchExpctd = [ 4 4 ];
            fpeExpctd = false;
            typExpctd = 'Type I';
            ordExpctd = [ 0 0 0 ];
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d();
            
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
            nchExpctd = [ 5 4 ];
            fpeExpctd = false;
            typExpctd = 'Type II';
            ordExpctd = [ 0 0 0 ];
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
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
            depth  = 16; 
            nch   = [ 4 4 ];
            ord   = [ 0 0 0 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ Ix(:) ; In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;            
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13);
            
        end           
        
        function testStepTypeII(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            nch   = [ 5 4 ];
            ord   = [ 0 0 0 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ Ix(:) ; In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13);
            
        end                   
        
        function testStepOrd222Ch44(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 2 2 ];
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [
                Ix(:) ; 
                In(:) ; 
                -In(:) ; 
                -In(:) ; 
                -In(:) ; 
                -In(:) ;                 
                -In(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13);
            
        end         
        
        function testStepOrd200Ch44(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 0 0 ];
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [
                Ix(:) ;
                In(:) ;
                -In(:) ;
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13);
            
        end
        
        function testStepOrd020Ch44(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 2 0 ];
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [
                Ix(:) ;
                In(:) ;
                -In(:) ;
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end
        
        function testStepOrd002Ch44(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 0 2 ];
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [
                Ix(:) ;
                In(:) ;
                -In(:) ;
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end        
        
        function testStepOrd220Ch44(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 2 0 ];
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [
                Ix(:) ;
                In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ];            
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13);
            
        end                
        
        function testStepOrd202Ch44(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 0 2 ];
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [
                Ix(:) ;
                In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ];            
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13);
            
        end                        
        
        function testStepOrd022Ch44(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 2 2 ];
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [
                Ix(:) ;
                In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ];            
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end
        
        function testStepOrd222Ch44H8W16D32(testCase)
            
            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 2 2 ];
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [
                Ix(:) ;
                In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;                
                -In(:) ;
                -In(:) ];            
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13);
            
        end                                        
        
        function testStepOrd422Ch44H8W16D32(testCase)
            
            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 4 2 2 ];
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [
                Ix(:) ;
                In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;                
                -In(:) ;
                -In(:) ;                
                -In(:) ;
                -In(:) ];            
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                
        
        function testStepOrd242Ch44H8W16D32(testCase)
            
            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 4 2 ];
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [
                Ix(:) ;
                In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;                
                -In(:) ;
                -In(:) ;                
                -In(:) ;
                -In(:) ];            
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                        
        
        function testStepOrd224Ch44H8W16D32(testCase)
            
            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 2 4 ];
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [
                Ix(:) ;
                In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;                
                -In(:) ;
                -In(:) ;                
                -In(:) ;
                -In(:) ];            
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                                
        
        function testStepOrd422Ch66H8W16D32U0(testCase)
            
            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 4 2 2 ];
            nch   = [ 6 6 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            U0 = dctmtx(nch(2));
            pmCoefs = [
                Ix(:) ;
                U0(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;
                -In(:) ;                
                -In(:) ;
                -In(:) ;                
                -In(:) ;
                -In(:) ];            
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            cfsExpctd(nch(1)+1:end,:) = U0.'*cfsExpctd(nch(1)+1:end,:);
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                                        
        
        function testStepOrd222Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 2 2 ];
            nch   = [ 5 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ 
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ; 
                Ix(:) ; 
                -In(:) ;                 
                Ix(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13);
            
        end           
        
        function testStepOrd200Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 0 0 ];
            nch   = [ 5 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ 
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end
        
        function testStepOrd020Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 2 0 ];
            nch   = [ 5 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ 
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                           
        
        function testStepOrd002Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 0 2 ];
            nch   = [ 5 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ 
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                   
        
        function testStepOrd220Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 2 0 ];
            nch   = [ 5 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ 
                Ix(:) ; 
                In(:) ;
                Ix(:) ; 
                -In(:) ;                
                Ix(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                   
        
        function testStepOrd022Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 2 2 ];
            nch   = [ 5 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ 
                Ix(:) ; 
                In(:) ;
                Ix(:) ; 
                -In(:) ;                
                Ix(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                           
        
        function testStepOrd202Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 0 2 ];
            nch   = [ 5 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ 
                Ix(:) ; 
                In(:) ;
                Ix(:) ; 
                -In(:) ;                
                Ix(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                   
        
        function testStepOrd222Ch54H8W16D32(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 2 2 ];
            nch   = [ 5 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ 
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ;                
                Ix(:) ; 
                -In(:) ;                                
                Ix(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                           
        
        function testStepOrd422Ch54H8W16D32(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 4 2 2 ];
            nch   = [ 5 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ 
                Ix(:) ; 
                In(:) ;
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ;                
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                                   
        
        function testStepOrd242Ch54H8W16D32(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 4 2 ];
            nch   = [ 5 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ 
                Ix(:) ; 
                In(:) ;
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ;                
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                                           
        
        function testStepOrd224Ch54H8W16D32(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 2 4 ];
            nch   = [ 5 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            pmCoefs = [ 
                Ix(:) ; 
                In(:) ;
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ;                
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                                                   
        
        function testStepOrd222Ch64H8W16D32U0(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 2 4 ];
            nch   = [ 6 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            In = eye(nch(2));
            U0 = dctmtx(nch(2));
            pmCoefs = [ 
                Ix(:) ; 
                U0(:) ;
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ;                
                Ix(:) ; 
                -In(:) ;
                Ix(:) ; 
                -In(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            cfsExpctd(nch(1)+1:end,:) = U0.'*cfsExpctd(nch(1)+1:end,:);
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end  
        
        function testStepOrd002Ch44RandAng(testCase)

            % Parameters
            height = 4;
            width  = 4;
            depth  = 4;
            ord   = [ 0 0 2 ];
            nch   = [ 4 4 ];
            nhch   = sum(nch)/2;
            arrayCoefs = repmat(1:height*width*depth,[sum(nch),1]);
            scale = [ height width depth ];
            
            %
            import saivdr.dictionary.utility.*            
            npm = 6;
            angs = randn(npm,2+sum(ord));
            mus  = ones(nch(1),2+sum(ord));
            omg = OrthonormalMatrixGenerationSystem();            
            W0  = step(omg,angs(:,1),mus(:,1));
            U0  = step(omg,angs(:,2),mus(:,2));
            Uz1 = step(omg,angs(:,3),mus(:,3));
            Uz2 = step(omg,angs(:,4),mus(:,4));        
            I = eye(nhch);
            B  = [ I I ; I -I ]/sqrt(2);
            %
            pmCoefs = [ 
                W0(:) ; 
                U0(:) ;
                Uz1(:) ; 
                Uz2(:) ];
            %
            R0  = blkdiag(W0.',U0.');
            Rz1 = blkdiag(I,Uz1.');
            Rz2 = blkdiag(I,Uz2.');
            coefs_ = B*Rz2*arrayCoefs;
            % right shift upper coefs
            tmp = coefs_(1:nhch,end-width*height+1:end);
            coefs_(1:nhch,width*height+1:end) = coefs_(1:nhch,1:end-width*height);
            coefs_(1:nhch,1:width*height) = tmp;
            %
            coefs_ = B*coefs_;
            coefs_ = B*Rz1*coefs_;
            % left shift lower coefs
            tmp = coefs_(nhch+1:end,1:width*height);
            coefs_(nhch+1:end,1:end-width*height) = coefs_(nhch+1:end,width*height+1:end);
            coefs_(nhch+1:end,end-width*height+1:end) = tmp;
            %            
            coefs_ = B*coefs_;            
            coefs_ = R0*coefs_;
          
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'IsPeriodicExt',true);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,arrayCoefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifySize(cfsActual,size(cfsExpctd));
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end
        
        function testStepOrd002Ch54RandAng(testCase)

            % Parameters
            height = 4;
            width  = 4;
            depth  = 4;
            ord   = [ 0 0 2 ];
            nch   = [ 5 4 ];
            nhx   = max(nch);
            nhn   = min(nch);
            arrayCoefs = repmat(1:height*width*depth,[sum(nch),1]);
            scale = [ height width depth ];
            
            %
            import saivdr.dictionary.utility.*            
            npmW = 10;
            npmU = 6;
            npm = npmW+npmU;
            angs = randn(npm,(2+sum(ord))/2);
            mus  = ones(sum(nch),(2+sum(ord))/2);
            omgW = OrthonormalMatrixGenerationSystem();            
            omgU = OrthonormalMatrixGenerationSystem();            
            W0  = step(omgW,angs(1:npmW,1),mus(1:nch(1),1));
            U0  = step(omgU,angs(npmW+1:end,1),mus(nch(1)+1:end,1));
            Wz1 = step(omgW,angs(1:npmW,2),mus(1:nch(1),2));
            Uz1 = step(omgU,angs(npmW+1:end,2),mus(nch(1)+1:end,2));        
            In = eye(nhn);
            Id = eye(nhx-nhn);
            zn = zeros(nhn,1);
            B  = [ In zn In ; 
                zn.' sqrt(2)*Id zn.' ;
                  In zn -In ]/sqrt(2);
            %
            pmCoefs = [ 
                W0(:) ; 
                U0(:) ;
                Wz1(:) ; 
                Uz1(:) ];
            %
            R0  = blkdiag(W0.',U0.');
            Rz1 = blkdiag(eye(nhx),Uz1.');
            Rz2 = blkdiag(Wz1.',eye(nhn));
            coefs_ = B*Rz2*arrayCoefs;
            % right shift upper coefs
            tmp = coefs_(1:nhx,end-width*height+1:end);
            coefs_(1:nhx,width*height+1:end) = coefs_(1:nhx,1:end-width*height);
            coefs_(1:nhx,1:width*height) = tmp;
            %
            coefs_ = B*coefs_;
            coefs_ = B*Rz1*coefs_;
            % left shift lower coefs
            tmp = coefs_(nhn+1:end,1:width*height);
            coefs_(nhn+1:end,1:end-width*height) = coefs_(nhn+1:end,width*height+1:end);
            coefs_(nhn+1:end,end-width*height+1:end) = tmp;
            %            
            coefs_ = B*coefs_;            
            coefs_ = R0*coefs_;
          
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'IsPeriodicExt',true);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,arrayCoefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifySize(cfsActual,size(cfsExpctd));
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end        
               
        function testStepOrd222Ch45(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 2 2 ];
            nch   = [ 4 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            Ix = eye(nch(2));
            pmCoefs = [ 
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ; 
                -In(:) ; 
                Ix(:) ;                 
                -In(:) ; 
                Ix(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13);
            
    end           
        
    function testStepOrd200Ch45(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 0 0 ];
            nch   = [ 4 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            Ix = eye(nch(2));
            pmCoefs = [ 
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end

        function testStepOrd020Ch45(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 2 0 ];
            nch   = [ 4 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            Ix = eye(nch(2));
            pmCoefs = [ 
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                           

        function testStepOrd002Ch45(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 0 2 ];
            nch   = [ 4 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            Ix = eye(nch(2));
            pmCoefs = [ 
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                   

        function testStepOrd220Ch45(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 2 0 ];
            nch   = [ 4 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            Ix = eye(nch(2));
            pmCoefs = [ 
                In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ;                
                -In(:) ; 
                Ix(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                   

        function testStepOrd022Ch45(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 2 2 ];
            nch   = [ 4 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            Ix = eye(nch(2));
            pmCoefs = [ 
                In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ;                
                -In(:) ; 
                Ix(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                           

        function testStepOrd202Ch45(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 0 2 ];
            nch   = [ 4 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            Ix = eye(nch(2));
            pmCoefs = [ 
                In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ;                
                -In(:) ; 
                Ix(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                   

        function testStepOrd222Ch45H8W16D32(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 2 2 ];
            nch   = [ 4 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            Ix = eye(nch(2));
            pmCoefs = [ 
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ;                
                -In(:) ; 
                Ix(:) ;                                
                -In(:) ; 
                Ix(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end

        function testStepOrd422Ch45H8W16D32(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 4 2 2 ];
            nch   = [ 4 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            Ix = eye(nch(2));
            pmCoefs = [ 
                In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ;                
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                                   

        function testStepOrd242Ch45H8W16D32(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 4 2 ];
            nch   = [ 4 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            Ix = eye(nch(2));
            pmCoefs = [ 
                In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ;                
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                                           

        function testStepOrd224Ch45H8W16D32(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 2 4 ];
            nch   = [ 4 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            Ix = eye(nch(2));
            pmCoefs = [ 
                In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ;                
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end                                                                                   

        function testStepOrd222Ch46H8W16D32U0(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 2 4 ];
            nch   = [ 4 6 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            Ix = eye(nch(2));
            W0 = dctmtx(nch(1));
            pmCoefs = [ 
                W0(:) ;
                Ix(:) ; 
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ;                
                -In(:) ; 
                Ix(:) ;
                -In(:) ; 
                Ix(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            cfsExpctd(1:nch(1),:) = W0.'*cfsExpctd(1:nch(1),:);
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end  

        function testStepOrd002Ch45RandAng(testCase)

            % Parameters
            height = 4;
            width  = 4;
            depth  = 4;
            ord   = [ 0 0 2 ];
            nch   = [ 4 5 ];
            nhx   = max(nch);
            nhn   = min(nch);
            arrayCoefs = repmat(1:height*width*depth,[sum(nch),1]);
            scale = [ height width depth ];
            
            %
            import saivdr.dictionary.utility.*            
            npmW = 6;
            npmU = 10;
            npm = npmW+npmU;
            angs = randn(npm,(2+sum(ord))/2);
            mus  = ones(sum(nch),(2+sum(ord))/2);
            omgW = OrthonormalMatrixGenerationSystem();            
            omgU = OrthonormalMatrixGenerationSystem();            
            W0  = step(omgW,angs(1:npmW,1),mus(1:nch(1),1));
            U0  = step(omgU,angs(npmW+1:end,1),mus(nch(1)+1:end,1));
            Wz1 = step(omgW,angs(1:npmW,2),mus(1:nch(1),2));
            Uz1 = step(omgU,angs(npmW+1:end,2),mus(nch(1)+1:end,2));        
            In = eye(nhn);
            Id = eye(nhx-nhn);
            zn = zeros(nhn,1);
            B  = [ In zn In ; 
                zn.' sqrt(2)*Id zn.' ;
                  In zn -In ]/sqrt(2);
            %
            pmCoefs = [ 
                W0(:) ; 
                U0(:) ;
                Wz1(:) ; 
                Uz1(:) ];
            %
            R0  = blkdiag(W0.',U0.');
            Rz1 = blkdiag(Wz1.',eye(nhx));
            Rz2 = blkdiag(eye(nhn),Uz1.');
            coefs_ = B*Rz2*arrayCoefs;
            % right shift upper coefs
            tmp = coefs_(1:nhx,end-width*height+1:end);
            coefs_(1:nhx,width*height+1:end) = coefs_(1:nhx,1:end-width*height);
            coefs_(1:nhx,1:width*height) = tmp;
            %
            coefs_ = B*coefs_;
            coefs_ = B*Rz1*coefs_;
            % left shift lower coefs
            tmp = coefs_(nhn+1:end,1:width*height);
            coefs_(nhn+1:end,1:end-width*height) = coefs_(nhn+1:end,width*height+1:end);
            coefs_(nhn+1:end,end-width*height+1:end) = tmp;
            %            
            coefs_ = B*coefs_;            
            coefs_ = R0*coefs_;
          
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.NsoltAtomConcatenator3d
            testCase.module = NsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'IsPeriodicExt',true);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,arrayCoefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifySize(cfsActual,size(cfsExpctd));
            diff = max(abs(cfsActual(:)-cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-13,...
                sprintf('diff = %e',diff));
            
        end        

    end
 
end
