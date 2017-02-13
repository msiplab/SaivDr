classdef CplxOLpPrFbAtomExtender1dTestCase < matlab.unittest.TestCase
    %OLPPRFBATOMEXTENDER1DTESTCASE Test case for CplxOLpPrFbAtomExtender1d
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
            ordExpctd = 0;
            fpeExpctd = false;
            typExpctd = 'Type I';
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d();
            
            % Actual values
            fpeActual = get(testCase.module,'IsPeriodicExt');
            nchActual = get(testCase.module,'NumberOfChannels');
            hchActual = get(testCase.module,'NumberOfHalfChannels');
            typActual = get(testCase.module,'OLpPrFbType');
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
            ordExpctd = 0;
            fpeExpctd = false;
            typExpctd = 'Type II';
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d(...
                'NumberOfChannels',nchExpctd);
            
            % Actual values
            fpeActual = get(testCase.module,'IsPeriodicExt');
            nchActual = get(testCase.module,'NumberOfChannels');
            hchActual = get(testCase.module,'NumberOfHalfChannels');
            typActual = get(testCase.module,'OLpPrFbType');
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
            srclen = 16;
            nch   = 4;
            ord   = 0;
            coefs = randn(nch, srclen) + 1i*randn(nch, srclen);
            scale = srclen;
            I0 = eye(nch);
            pmCoefs = I0(:);
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d(...
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
            srclen = 16;
            ord   = 0;
            nch   = 5;
            coefs = randn(nch, srclen) + 1i*randn(nch, srclen);
            scale = srclen;
            I0 = eye(nch);
            pmCoefs = I0(:);
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d(...
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
        
        function testStepOrd2Ch4(testCase)
            
            % Parameters
            %srclen = 16;
            srclen = 4;
            ord   = 2;
            nch   = 4;
            coefs = randn(nch, srclen) + 1i*randn(nch, srclen);
            scale = srclen;
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            angB = zeros(floor(nch/4),1);
            pmCoefs = [ I0(:) ; Ix(:) ; -In(:) ; angB ; In(:) ; -In(:) ; angB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d(...
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
        
        function testStepOrd2Ch4U0(testCase)
            
            % Parameters
            srclen = 16;
            ord   = 2;
            nch   = 4;
            coefs = randn(nch, srclen) + 1i*randn(nch, srclen);
            scale = srclen;
            V0 = dctmtx(nch);
            In = eye(floor(nch/2));
            angB = zeros(floor(nch/4),1);
            pmCoefs = [ V0(:) ; In(:) ; -In(:) ; angB; In(:) ; -In(:) ; angB];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = V0*coefs;
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d(...
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
        
        function testStepOrd4Ch4(testCase)
            
            % Parameters
            srclen = 16;
            ord   = 4;
            nch   = 4;
            coefs = randn(nch, srclen) + 1i*randn(nch, srclen);
            scale = srclen;
            I0 = eye(nch);
            In = eye(floor(nch/2));
            angB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ;
                -In(:) ;
                angB ;
                In(:) ;
                -In(:) ;
                angB(:) ;
                In(:) ;
                -In(:) ;
                angB(:) ;
                In(:) ;
                -In(:) ;
                angB(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d(...
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
        
        function testStepOrd2Ch8(testCase)
            
            % Parameters
            srclen = 16;
            ord   = 2;
            nch   = 8;
            coefs = randn(nch, srclen) + 1i*randn(nch, srclen);
            scale = srclen;
            I0 = eye(nch);
            In = eye(floor(nch/2));
            angB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ;
                -In(:) ;
                angB(:) ;
                In(:) ;
                -In(:) ;
                angB(:) ;
                In(:) ;
                -In(:) ;
                angB(:) ;
                In(:) ;
                -In(:) ;
                angB(:) ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d(...
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
        
        function testStepOrd2Ch5(testCase)
            
            % Parameters
            srclen = 16;
            ord   = 2;
            nch   = 5;
            coefs = randn(nch, srclen) + 1i*randn(nch, srclen);
            scale = srclen;
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Ux = blkdiag(-In,1);
            angB = zeros(floor(nch/4),1);
            pmCoefs = [ I0(:) ; In(:); -In(:); angB; Ix(:); Ux(:) ; angB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d(...
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
        
        function testStepOrd2Ch5U0(testCase)
            
            % Parameters
            srclen = 16;
            ord   = 2;
            nch   = 5;
            coefs = randn(nch, srclen) + 1i*randn(nch, srclen);
            scale = srclen;
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Ux = blkdiag(-In,1);
            V0 = dctmtx(nch);
            angB = zeros(floor(nch/4),1);
            pmCoefs = [ V0(:) ; In(:) ; -In(:) ; angB ; Ix(:) ; Ux(:) ; angB];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = V0*coefs;
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d(...
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
        
        function testStepOrd0Ch5(testCase)
            
            % Parameters
            srclen = 16;
            ord   = 0;
            nch   = 5;
            coefs = randn(nch, srclen) + 1i*randn(nch, srclen);
            scale = srclen;
            I0 = eye(nch);
            pmCoefs = I0(:);
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d(...
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
        
        function testStepOrd4Ch5(testCase)
            
            % Parameters
            srclen = 16;
            ord   = 4;
            nch   = 5;
            coefs = randn(nch, srclen) + 1i*randn(nch, srclen);
            scale = srclen;
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Ux = blkdiag(-In,1);
            angB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ;
                -In(:) ;
                angB ;
                Ix(:) ;
                Ux(:) ;
                angB ;
                In(:) ;
                -In(:) ;
                angB ;
                Ix(:) ;
                Ux(:) ;
                angB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d(...
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
        
        function testStepOrd2Ch9(testCase)
            
            % Parameters
            srclen = 16;
            ord   = 2;
            nch   = 9;
            coefs = randn(nch, srclen) + 1i*randn(nch, srclen);
            scale = srclen;
            I0 = eye(nch);
            Ix = eye(ceil(nch/2));
            In = eye(floor(nch/2));
            Ux = blkdiag(-In,1);
            angB = zeros(floor(nch/4),1);
            pmCoefs = [
                I0(:) ;
                In(:) ;
                -In(:) ;
                angB ;
                Ix(:) ;
                Ux(:) ;
                angB ];
            
            % Expected values
            ordExpctd = ord;
            cfsExpctd = coefs;
            
            % Instantiation
            import saivdr.dictionary.colpprfb.CplxOLpPrFbAtomExtender1d
            testCase.module = CplxOLpPrFbAtomExtender1d(...
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
