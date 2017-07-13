classdef NsoltVQStep2dTestCase < matlab.unittest.TestCase
    %NSOLTVQSTEP2DTESTCASE Test case for ModuleBlockDct2d
    %
    % SVN identifier:
    % $Id: NsoltVQStep2dTestCase.m 866 2015-11-24 04:29:42Z sho $
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015, Shogo MURAMATSU
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
            nchExpctd = [ 2 2 ];
            fpeExpctd = false;
            typExpctd = 'Type I';
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d();
            
            % Actual values
            fpeActual = get(testCase.module,'IsPeriodicExt');
            nchActual = [
                get(testCase.module,'NumberOfSymmetricChannels') ...
                get(testCase.module,'NumberOfAntisymmetricChannels') ];
            typActual = get(testCase.module,'NsoltType');
            
            % Evaluation
            testCase.verifyEqual(fpeActual,fpeExpctd);
            testCase.verifyEqual(nchActual,nchExpctd);
            testCase.verifyEqual(typActual,typExpctd);
            
        end
        
        function testStepCh22Idx1(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            In = eye(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 1;
            
            % Expected values
            cfsExpctd = blkdiag(W0,In)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx2(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            Ix = eye(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 2;
            
            % Expected values
            cfsExpctd = blkdiag(Ix,U0)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx3PeriodicExt(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 1 1 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            Ux = randn(nch(2));
            U2 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; Ux(:) ; U2(:) ];
            idx = 3;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt);
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx3(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 1 1 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            Ux = randn(nch(2));
            U2 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; Ux(:) ; U2(:) ];
            idx = 3;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            I = eye(size(Ux));
            for iCol = 1:width
                if iCol == 1 %&& ~isPeriodicExt
                    U = -I;
                else
                    U = Ux;
                end
                coefs_ = lowerBlockRot_(testCase,coefs_,iCol,U,nch,height);
            end
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt);
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx4PeriodicExt(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 1 1 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            Ux  = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; Ux(:) ];
            idx = 4;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt);
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx4(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 1 1 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            Ux  = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; Ux(:) ];
            idx = 4;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt);
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx5PeriodicExt(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ];
            idx = 5;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt);
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx5(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ];
            idx = 5;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            I = eye(size(Ux));
            for iCol = 1:width
                if iCol == 1 %&& ~isPeriodicExt
                    U = -I;
                else
                    U = Ux;
                end
                coefs_ = lowerBlockRot_(testCase,coefs_,iCol,U,nch,height);
            end
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt);
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx6PeriodicExt(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ];
            idx = 6;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt);
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx6(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ];
            idx = 6;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt);
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh44Idx5PeriodicExt(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            nch   = [ 4 4 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ];
            idx = 5;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt);
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh44Idx1(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            In = eye(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 1;
            
            % Expected values
            cfsExpctd = blkdiag(W0,In)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh44Idx2(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            Ix = eye(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 2;
            
            % Expected values
            cfsExpctd = blkdiag(Ix,U0)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh44Idx5(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            nch   = [ 4 4 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ];
            idx = 5;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            I = eye(size(Ux));
            for iCol = 1:width
                if iCol == 1 %&& ~isPeriodicExt
                    U = -I;
                else
                    U = Ux;
                end
                coefs_ = lowerBlockRot_(testCase,coefs_,iCol,U,nch,height);
            end
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt);
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh44Idx6PeriodicExt(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            nch   = [ 4 4 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ];
            idx = 6;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt);
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh44Idx6(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            nch   = [ 4 4 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ];
            idx = 6;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt);
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx1PartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            Zn = zeros(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 1;
            
            % Expected values
            cfsExpctd = blkdiag(W0,Zn)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx2PartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            Zx = zeros(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 2;
            
            % Expected values
            cfsExpctd = blkdiag(Zx,U0)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx3PeriodicExtPartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 1 1 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            Ux = randn(nch(2));
            U2 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; Ux(:) ; U2(:) ];
            idx = 3;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt,...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx3PartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 1 1 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            Ux = randn(nch(2));
            U2 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; Ux(:) ; U2(:) ];
            idx = 3;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            Z = zeros(size(Ux));
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            for iCol = 1:width
                if iCol == 1 %&& ~isPeriodicExt
                    U = Z;
                else
                    U = Ux;
                end
                coefs_ = lowerBlockRot_(testCase,coefs_,iCol,U,nch,height);
            end
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt,...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end

        function testStepCh22Idx4PeriodicExtPartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 1 1 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            Ux  = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; Ux(:) ];
            idx = 4;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt,...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx4PartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 1 1 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            Ux  = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; Ux(:) ];
            idx = 4;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt,...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end

        function testStepCh22Idx5PeriodicExtPartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ];
            idx = 5;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);            
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt,...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh22Idx5PartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ];
            idx = 5;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            Z = zeros(size(Ux));
            coefs_(1:nch(1),:) = 0*coefs_(1:nch(1),:);
            for iCol = 1:width
                if iCol == 1 %&& ~isPeriodicExt
                    U = Z;
                else
                    U = Ux;
                end
                coefs_ = lowerBlockRot_(testCase,coefs_,iCol,U,nch,height);
            end
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt,...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end

        function testStepCh22Idx6PeriodicExtPartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ];
            idx = 6;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt,...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
       
        function testStepCh22Idx6PartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            nch   = [ 2 2 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ];
            idx = 6;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);            
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt,...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end

        function testStepCh44Idx5PeriodicExtPartialDifference(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            nch   = [ 4 4 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ];
            idx = 5;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);            
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt,...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
                
        function testStepCh44Idx1PartialDifference(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            Zn = zeros(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 1;
            
            % Expected values
            cfsExpctd = blkdiag(W0,Zn)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh44Idx2PartialDifference(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            Zx = zeros(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 2;
            
            % Expected values
            cfsExpctd = blkdiag(Zx,U0)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh44Idx5PartialDifference(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            nch   = [ 4 4 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ];
            idx = 5;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            Z = zeros(size(Ux));
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);                        
            for iCol = 1:width
                if iCol == 1 %&& ~isPeriodicExt
                    U = Z;
                else
                    U = Ux;
                end
                coefs_ = lowerBlockRot_(testCase,coefs_,iCol,U,nch,height);
            end
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt,...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
       
        function testStepCh44Idx6PeriodicExtPartialDifference(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            nch   = [ 4 4 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ];
            idx = 6;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt,...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh44Idx6PartialDifference(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            nch   = [ 4 4 ];
            ord   = [ 2 2 ];
            coefs = randn(sum(nch), height*width);
            scale = [ height width ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ];
            idx = 6;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep2d
            testCase.module = NsoltVQStep2d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2),...
                'PolyPhaseOrder',ord,...
                'IsPeriodicExt',isPeriodicExt,...
                'PartialDifference','on');
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
    end
    
    methods (Access = private)
        
        function arrayCoefs = rightShiftLowerCoefs_(~,arrayCoefs,nch,nRows_)
            hLenMn = max(nch);
            %
            lowerCoefsPre = arrayCoefs(hLenMn+1:end,end-nRows_+1:end);
            arrayCoefs(hLenMn+1:end,nRows_+1:end) = ...
                arrayCoefs(hLenMn+1:end,1:end-nRows_);
            arrayCoefs(hLenMn+1:end,1:nRows_) = ...
                lowerCoefsPre;
        end
        
        function arrayCoefs = leftShiftUpperCoefs_(~,arrayCoefs,nch,nRows_)
            hLenMx = min(nch);
            %
            upperCoefsPost = arrayCoefs(1:hLenMx,1:nRows_);
            arrayCoefs(1:hLenMx,1:end-nRows_) = ...
                arrayCoefs(1:hLenMx,nRows_+1:end);
            arrayCoefs(1:hLenMx,end-nRows_+1:end) = ...
                upperCoefsPost;
        end
        
        function arrayCoefs = blockButterflyTypeI_(~,arrayCoefs,nch)
            hLen = nch(1);
            upper = arrayCoefs(1:hLen,:);
            lower = arrayCoefs(hLen+1:end,:);
            arrayCoefs = [
                upper + lower;
                upper - lower ];
        end
        
        function arrayCoefs = lowerBlockRot_(~,arrayCoefs,iCol,U,nch,nRows_)
            hLen = nch(1);
            indexCol = (iCol-1)*nRows_;
            arrayCoefs(hLen+1:end,indexCol+1:indexCol+nRows_) = ...
                U*arrayCoefs(hLen+1:end,indexCol+1:indexCol+nRows_);
        end
        
    end
end
