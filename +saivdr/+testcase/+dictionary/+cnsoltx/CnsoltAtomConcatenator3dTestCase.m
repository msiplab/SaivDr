classdef CnsoltAtomConcatenator3dTestCase < matlab.unittest.TestCase
    %MODULEBLOCKDCT3dTESTCASE Test case for ModuleBlockDct3d
    %
    % SVN identifier:
    % $Id: CnsoltAtomConcatenator3dTestCase.m 683 2015-05-29 08:22:13Z sho $
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
            nchExpctd = 8;
            hchExpctd = 4;
            fpeExpctd = false;
            typExpctd = 'Type I';
            ordExpctd = [ 0 0 0 ];
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d();
            
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
            nchExpctd = 9;
            hchExpctd = 4;
            fpeExpctd = false;
            typExpctd = 'Type II';
            ordExpctd = [ 0 0 0 ];
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d('NumberOfChannels',nchExpctd);
            
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
            depth  = 16; 
            nch   = 8;
            ord   = [ 0 0 0 ];
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch);
            pmCoefs = Ix(:);
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;            
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8);
            
        end           
        
        function testStepTypeII(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            nch   = 9;
            ord   = [ 0 0 0 ];
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch);
            pmCoefs = Ix(:);
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8);
            
        end                   
        
        function testStepOrd222Ch44(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 2 2 ];
            nch   = 8;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8);
            
        end         
        
        function testStepOrd200Ch44(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 0 0 ];
            nch   = 8;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8);
            
        end
        
        function testStepOrd020Ch44(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 2 0 ];
            nch   = 8;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end
        
        function testStepOrd002Ch44(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 0 2 ];
            nch   = 8;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end        
        
        function testStepOrd220Ch44(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 2 0 ];
            nch   = 8;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ];          
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8);
            
        end                
        
        function testStepOrd202Ch44(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 0 2 ];
            nch   = 8;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ];        
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8);
            
        end                        
        
        function testStepOrd022Ch44(testCase)
            
            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 2 2 ];
            nch   = 8;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ];          
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end
        
        function testStepOrd222Ch44H8W16D32(testCase)
            
            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 2 2 ];
            nch   = 8;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ];           
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8);
            
        end                                        
        
        function testStepOrd422Ch44H8W16D32(testCase)
            
            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 4 2 2 ];
            nch   = 8;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end                                                
        
        function testStepOrd242Ch44H8W16D32(testCase)
            
            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 4 2 ];
            nch   = 8;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ];       
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end                                                        
        
        function testStepOrd224Ch44H8W16D32(testCase)
            
            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 2 4 ];
            nch   = 8;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ];          
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end                                                                
        
        function testStepOrd422Ch66H8W16D32U0(testCase)
            
            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 4 2 2 ];
            nch   = 12;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            V0 = dctmtx(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                V0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ; -Ix(:) ; angsB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            cfsExpctd = V0.'*cfsExpctd;
            
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end                                                                        
        
        function testStepOrd222Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 2 2 ];
            nch   = 9;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Is = blkdiag(-In,1);
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ];  
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8);
            
        end           
        
        function testStepOrd200Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 0 0 ];
            nch   = 9;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Is = blkdiag(-In,1);
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ];  
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end
        
        function testStepOrd020Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 2 0 ];
            nch   = 9;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Is = blkdiag(-In,1);
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ];  
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end                           
        
        function testStepOrd002Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 0 2 ];
            nch   = 9;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Is = blkdiag(-In,1);
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ];  
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end                                   
        
        function testStepOrd220Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 2 0 ];
            nch   = 9;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Is = blkdiag(-In,1);
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end                                   
        
        function testStepOrd022Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 0 2 2 ];
            nch   = 9;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Is = blkdiag(-In,1);
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ];  
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end                                           
        
        function testStepOrd202Ch54(testCase)

            % Parameters
            height = 16;
            width  = 16;
            depth  = 16;
            ord   = [ 2 0 2 ];
            nch   = 9;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Is = blkdiag(-In,1);
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ];  
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end                                                   
        
        function testStepOrd222Ch54H8W16D32(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 2 2 ];
            nch   = 9;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Is = blkdiag(-In,1);
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ];  
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end                                                           
        
        function testStepOrd422Ch54H8W16D32(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 4 2 2 ];
            nch   = 9;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Is = blkdiag(-In,1);
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end                                                                   
        
        function testStepOrd242Ch54H8W16D32(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 4 2 ];
            nch   = 9;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Is = blkdiag(-In,1);
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end                                                                           
        
        function testStepOrd224Ch54H8W16D32(testCase)

            % Parameters
            height = 8;
            width  = 16;
            depth  = 32;
            ord   = [ 2 2 4 ];
            nch   = 9;
            coefs = randn(nch, height*width*depth);
            scale = [ height width depth ];
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Is = blkdiag(-In,1);
            angsB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ;
                In(:) ; -In(:) ; angsB ;
                Ix(:) ;  Is(:) ; angsB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
                        
            % Instantiation
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfChannels',nch);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,coefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end
        
        function testStepOrd002Ch44RandAng(testCase)

            % Parameters
            height = 4;
            width  = 4;
            depth  = 4;
            ord   = [ 0 0 2 ];
            nch   = 8;
            nhch   = nch/2;
            arrayCoefs = repmat(1:height*width*depth,[nch,1]);
            scale = [ height width depth ];
            
            %
            import saivdr.dictionary.utility.*            
            npm = 6;
            angs = randn(npm,2+sum(ord));
            mus  = ones(ceil(nch/2),2+sum(ord));
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
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',ceil(nch/2),...
                'NumberOfAntisymmetricChannels',floor(nch/2),...
                'IsPeriodicExt',true);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,arrayCoefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifySize(cfsActual,size(cfsExpctd));
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end
        
        function testStepOrd002Ch54RandAng(testCase)

            % Parameters
            height = 4;
            width  = 4;
            depth  = 4;
            ord   = [ 0 0 2 ];
            nch   = 9;
            nhx   = max(nch);
            nhn   = min(nch);
            arrayCoefs = repmat(1:height*width*depth,[nch,1]);
            scale = [ height width depth ];
            
            %
            import saivdr.dictionary.utility.*            
            npmW = 10;
            npmU = 6;
            npm = npmW+npmU;
            angs = randn(npm,(2+sum(ord))/2);
            mus  = ones(nch,(2+sum(ord))/2);
            omgW = OrthonormalMatrixGenerationSystem();            
            omgU = OrthonormalMatrixGenerationSystem();            
            W0  = step(omgW,angs(1:npmW,1),mus(1:ceil(nch/2),1));
            U0  = step(omgU,angs(npmW+1:end,1),mus(ceil(nch/2)+1:end,1));
            Wz1 = step(omgW,angs(1:npmW,2),mus(1:ceil(nch/2),2));
            Uz1 = step(omgU,angs(npmW+1:end,2),mus(ceil(nch/2)+1:end,2));        
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
            import saivdr.dictionary.cnsoltx.CnsoltAtomConcatenator3d
            testCase.module = CnsoltAtomConcatenator3d(...
                'NumberOfSymmetricChannels',ceil(nch/2),...
                'NumberOfAntisymmetricChannels',floor(nch/2),...
                'IsPeriodicExt',true);
            set(testCase.module,'PolyPhaseOrder',ord);
            
            % Actual values
            ordActual = get(testCase.module,'PolyPhaseOrder');
            cfsActual = step(testCase.module,arrayCoefs,scale,pmCoefs);
            
            % Evaluation
            testCase.verifyEqual(ordActual,ordExpctd);
            testCase.verifySize(cfsActual,size(cfsExpctd));
            diff = max(abs(cfsActual(:)-cfsExpctd(:))./abs(cfsExpctd(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'RelTol',1e-8,...
                sprintf('diff = %e',diff));
            
        end
        
    end
 
end
