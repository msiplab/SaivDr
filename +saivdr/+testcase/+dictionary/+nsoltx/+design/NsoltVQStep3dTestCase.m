classdef NsoltVQStep3dTestCase < matlab.unittest.TestCase
    %NSOLTVQSTEP3DTESTCASE Test case for ModuleBlockDct3d
    %
    % SVN identifier:
    % $Id: NsoltVQStep3dTestCase.m 866 2015-11-24 04:29:42Z sho $
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
            nchExpctd = [ 4 4 ];
            fpeExpctd = false;
            typExpctd = 'Type I';
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d();
            
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

        function testStepCh44Idx1(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth];
            In = eye(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 1;
            
            % Expected values
            cfsExpctd = blkdiag(W0,In)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth];
            Ix = eye(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 2;
            
            % Expected values
            cfsExpctd = blkdiag(Ix,U0)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh44Idx3PeriodicExt(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 1 1 1 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            Ux = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; Ux(:) ; U2(:); U3(:) ];
            idx = 3;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
        
        function testStepCh44Idx3(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 1 1 1 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            Ux = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));            
            pmCoefs = [ W0(:) ; U0(:) ; Ux(:) ; U2(:) ; U3(:) ];
            idx = 3;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            I = eye(size(Ux));
            for iLay = 1:depth
                if iLay == 1 %&& ~isPeriodicExt
                    U = -I;
                else
                    U = Ux;
                end
                coefs_ = lowerBlockRot_(testCase,coefs_,iLay,U,nch,height*width);
            end
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
        
       function testStepCh44Idx4PeriodicExt(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 1 1 1 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            Ux  = randn(nch(2));
            U3 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; Ux(:) ; U3(:) ];
            idx = 4;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
        
       function testStepCh44Idx4(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 1 1 1 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            Ux = randn(nch(2));
            U3 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; Ux(:) ; U3(:) ];
            idx = 4;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 1 1 1 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ];
            idx = 5;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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

        function testStepCh44Idx5(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 1 1 1 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ];
            idx = 5;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            I = eye(size(Ux));
            for iLay = 1:depth
                if iLay == 1 %&& ~isPeriodicExt
                    U = -I;
                else
                    U = Ux;
                end
                coefs_ = lowerBlockRot_(testCase,coefs_,iLay,U,nch,height*width);
            end
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));            
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ; U5(:) ; U6(:) ];
            idx = 6;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));            
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ; U5(:) ; U6(:) ];
            idx = 6;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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

        function testStepCh55Idx5PeriodicExt(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            depth  = 40;
            nch   = [ 5 5 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));            
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ; U5(:) ; U6(:) ];
            idx = 5;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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

        function testStepCh55Idx1(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            depth  = 40;
            nch   = [ 5 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            In = eye(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 1;
            
            % Expected values
            cfsExpctd = blkdiag(W0,In)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end

        function testStepCh55Idx2(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            depth  = 40;
            nch   = [ 5 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Ix = eye(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 2;
            
            % Expected values
            cfsExpctd = blkdiag(Ix,U0)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
                'NumberOfSymmetricChannels',nch(1),...
                'NumberOfAntisymmetricChannels',nch(2));
            
            % Actual values
            cfsActual = step(testCase.module,coefs,scale,pmCoefs,idx);
            
            % Evaluation
            diff = max(abs(cfsExpctd(:)-cfsActual(:)));
            testCase.verifyEqual(cfsActual,cfsExpctd,'AbsTol',1e-10,...
                sprintf('diff = %f',diff));
            
        end
        
        function testStepCh55Idx5(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            depth  = 40;
            nch   = [ 5 5 ];
            ord   = [ 2 2 3 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ; U5(:) ; U6(:) ];
            idx = 5;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            I = eye(size(Ux));
            for iLay = 1:depth
                if iLay == 1 %&& ~isPeriodicExt
                    U = -I;
                else
                    U = Ux;
                end
                coefs_ = lowerBlockRot_(testCase,coefs_,iLay,U,nch,height*width);
            end
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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

        function testStepCh55Idx6PeriodicExt(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            depth  = 40;
            nch   = [ 5 5 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ; U5(:) ; U6(:) ];
            idx = 6;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
        
        function testStepCh55Idx6(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            depth  = 40;
            nch   = [ 4 4 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ; U5(:) ; U6(:) ];
            idx = 6;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
        
        function testStepCh44Idx1PartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Zn = zeros(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 1;
            
            % Expected values
            cfsExpctd = blkdiag(W0,Zn)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Zx = zeros(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 2;
            
            % Expected values
            cfsExpctd = blkdiag(Zx,U0)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
        
        function testStepCh44Idx3PeriodicExtPartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 1 1 1 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            Ux = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; Ux(:) ; U2(:) ; U3(:) ];
            idx = 3;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
        
        function testStepCh44Idx3PartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 1 1 1 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            Ux = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; Ux(:) ; U2(:) ; U3(:) ];
            idx = 3;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            Z = zeros(size(Ux));
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            for iLay = 1:depth
                if iLay == 1 %&& ~isPeriodicExt
                    U = Z;
                else
                    U = Ux;
                end
                coefs_ = lowerBlockRot_(testCase,coefs_,iLay,U,nch,height*width);
            end
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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

        function testStepCh44Idx4PeriodicExtPartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 1 1 1 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            Ux = randn(nch(2));
            U3 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; Ux(:) ; U3(:) ];
            idx = 4;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
        
        function testStepCh44Idx4PartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 1 1 1 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            Ux = randn(nch(2));
            U3 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; Ux(:) ; U3(:) ];
            idx = 4;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            U5 = randn(nch(2));
            U6= randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ; U5(:) ; U6(:) ];
            idx = 5;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);            
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
        
        function testStepCh44Idx5PartialDifference(testCase)
            
            % Parameters
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ; U5(:) ; U6(:) ];
            idx = 5;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            Z = zeros(size(Ux));
            coefs_(1:nch(1),:) = 0*coefs_(1:nch(1),:);
            for iLay = 1:depth
                if iLay == 1 %&& ~isPeriodicExt
                    U = Z;
                else
                    U = Ux;
                end
                coefs_ = lowerBlockRot_(testCase,coefs_,iLay,U,nch,height*width);
            end
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux  = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));            
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ; U5(:) ; U6(:) ];
            idx = 6;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
            height = 12;
            width  = 16;
            depth  = 20;
            nch   = [ 4 4 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ; U5(:) ; U6(:) ];
            idx = 6;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);            
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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

        function testStepCh55Idx5PeriodicExtPartialDifference(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            depth  = 40;
            nch   = [ 5 5 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ; U5(:) ; U6(:) ];
            idx = 5;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);            
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
           
        function testStepCh55Idx1PartialDifference(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            depth  = 40;
            nch   = [ 5 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Zn = zeros(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 1;
            
            % Expected values
            cfsExpctd = blkdiag(W0,Zn)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
        
        function testStepCh55Idx2PartialDifference(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            depth  = 40;
            nch   = [ 5 5 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            Zx = zeros(nch(1));
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            pmCoefs = [ W0(:) ; U0(:) ];
            idx = 2;
            
            % Expected values
            cfsExpctd = blkdiag(Zx,U0)*coefs;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
        
        function testStepCh55Idx5PartialDifference(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            depth  = 40;
            nch   = [ 5 5 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            Ux = randn(nch(2));
            U4 = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));            
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; Ux(:) ; U4(:) ; U5(:) ; U6(:) ];
            idx = 5;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = rightShiftLowerCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            Z = zeros(size(Ux));
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);                        
            for iLay = 1:depth
                if iLay == 1 %&& ~isPeriodicExt
                    U = Z;
                else
                    U = Ux;
                end
                coefs_ = lowerBlockRot_(testCase,coefs_,iLay,U,nch,height*width);
            end
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
       
        function testStepCh55Idx6PeriodicExtPartialDifference(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            depth  = 40;
            nch   = [ 5 5 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));            
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ; U5(:) ; U6(:) ];
            idx = 6;
            isPeriodicExt = true;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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

        function testStepCh55Idx6PartialDifference(testCase)
            
            % Parameters
            height = 24;
            width  = 32;
            depth  = 40;
            nch   = [ 5 5 ];
            ord   = [ 2 2 2 ];
            coefs = randn(sum(nch), height*width*depth);
            scale = [ height width depth ];
            W0 = randn(nch(1));
            U0 = randn(nch(2));
            U1 = randn(nch(2));
            U2 = randn(nch(2));
            U3 = randn(nch(2));
            Ux = randn(nch(2));
            U5 = randn(nch(2));
            U6 = randn(nch(2));            
            pmCoefs = [ W0(:) ; U0(:) ; U1(:) ; U2(:) ; U3(:) ; Ux(:) ; U5(:) ; U6(:) ];
            idx = 6;
            isPeriodicExt = false;
            
            % Expected values
            %I = eye(size(U1));
            coefs_ = blockButterflyTypeI_(testCase,coefs,nch);
            coefs_ = leftShiftUpperCoefs_(testCase,coefs_,nch,height*width);
            coefs_ = blockButterflyTypeI_(testCase,coefs_,nch);
            coefs_ = coefs_/2.0;
            % Lower channel rotation
            coefs_(1:nch(1),:)     = 0*coefs_(1:nch(1),:);
            coefs_(nch(1)+1:end,:) = Ux*coefs_(nch(1)+1:end,:);
            cfsExpctd = coefs_;
            
            % Instantiation
            import saivdr.dictionary.nsoltx.design.NsoltVQStep3d
            testCase.module = NsoltVQStep3d(...
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
        
        function arrayCoefs = rightShiftLowerCoefs_(~,arrayCoefs,nch,nRowsxnCols_)
            hLenMn = max(nch);
            %
            lowerCoefsPre = arrayCoefs(hLenMn+1:end,end-nRowsxnCols_+1:end);
            arrayCoefs(hLenMn+1:end,nRowsxnCols_+1:end) = ...
                arrayCoefs(hLenMn+1:end,1:end-nRowsxnCols_);
            arrayCoefs(hLenMn+1:end,1:nRowsxnCols_) = ...
                lowerCoefsPre;
        end
        
        function arrayCoefs = leftShiftUpperCoefs_(~,arrayCoefs,nch,nRowsxnCols_)
            hLenMx = min(nch);
            %
            upperCoefsPost = arrayCoefs(1:hLenMx,1:nRowsxnCols_);
            arrayCoefs(1:hLenMx,1:end-nRowsxnCols_) = ...
                arrayCoefs(1:hLenMx,nRowsxnCols_+1:end);
            arrayCoefs(1:hLenMx,end-nRowsxnCols_+1:end) = ...
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
        
        function arrayCoefs = lowerBlockRot_(~,arrayCoefs,iLay,U,nch,nRowsxnCols_)
            hLen = nch(1);
            indexLay = (iLay-1)*nRowsxnCols_;
            arrayCoefs(hLen+1:end,indexLay+1:indexLay+nRowsxnCols_) = ...
                U*arrayCoefs(hLen+1:end,indexLay+1:indexLay+nRowsxnCols_);
        end
        
    end
end
