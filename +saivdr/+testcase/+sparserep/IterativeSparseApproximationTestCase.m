classdef IterativeSparseApproximationTestCase < matlab.unittest.TestCase
    % ITERATIVESPARSERESTORATIONTESTCASE Test case for IterativeSparseApproximation
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2018, Shogo MURAMATSU
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
    
    properties (TestParameter)
        width = struct('small', 8, 'large', 16);
        height = struct('small', 8, 'large', 16);
        depth = struct('small', 8,  'large', 16);
        vdec = struct('small', 1,  'large', 2);
        hdec = struct('small', 1,  'large', 2);
        ddec = struct('small', 1,  'large', 2);
        sch = struct('small', 2,  'large', 4);
        ach = struct('small', 2,  'large', 4);
    end
    
    properties
        target
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.target);
        end
    end
    
    methods (Test)
        
        function testConstruction(testCase)
            
            % Expected Value
            algorithmExpctd = [];
            
            % Instantiation
            import saivdr.sparserep.IterativeSparseApproximater
            testCase.target = IterativeSparseApproximater();
            
            % Actual Value
            algorithmActual = testCase.target.Algorithm;
            
            % Evaluation
            testCase.verifyEqual(algorithmActual,algorithmExpctd);
        end
        
        function testStepIsta2dSize(testCase,height,width)
            
            nDecs = [ 2 2 ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            nVms = 1;
            wSigma = 1e-1;
            lambda = 1e-4;
            srcImg = imgaussfilt(rand(height,width),4);
            obsImg = srcImg + wSigma*randn(height,width);
            
            % Expected values
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', nVms,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            lppufb.Angles = angs;
            
            % Instantiation of target class
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            itermethod = IstaSystem('Lambda',lambda);
            testCase.target = IterativeSparseApproximater(...
                'Algorithm',itermethod,...
                'Dictionary', {synthesizer,analyzer});
            
            % Actual values
            resActual = testCase.target.step(obsImg);
            
            % Evaluation
            testCase.verifySize(resActual, size(srcImg));
            
            import matlab.unittest.constraints.IsLessThan
            rmse = @(x,y) norm(x(:)-y(:))/sqrt(numel(x));
            rmsePre = rmse(srcImg,obsImg);
            rmsePst = rmse(srcImg,resActual);
            testCase.verifyThat(rmsePst,IsLessThan(rmsePre));
            
        end
        
        function testStepIsta3dSize(testCase,height,width,depth)
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 2 ];
            nVms  = 1;
            wSigma = 1e-1;
            lambda = 1e-4;
            srcImg = imgaussfilt3(rand(height,width,depth),4);
            obsImg = srcImg + wSigma*randn(height,width,depth);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', nVms,...                
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            lppufb.Angles = angs;
            
            % Instantiation of target class
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            itermethod = IstaSystem('Lambda',lambda);
            testCase.target = IterativeSparseApproximater(...
                'Algorithm',itermethod,...
                'Dictionary', {synthesizer,analyzer});
            
            % Actual values
            resActual = testCase.target.step(obsImg);
            
            % Evaluation
            testCase.verifySize(resActual, size(srcImg));
            
            import matlab.unittest.constraints.IsLessThan
            rmse = @(x,y) norm(x(:)-y(:))/sqrt(numel(x));
            rmsePre = rmse(srcImg,obsImg);
            rmsePst = rmse(srcImg,resActual);
            testCase.verifyThat(rmsePst,IsLessThan(rmsePre));
            
        end
            
        function testStepIsta2dDecimation(testCase,...
                vdec,hdec)
            
            nDecs = [ vdec hdec ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            nVms = 1;
            height_ = 16;
            width_  = 16;
            wSigma = 1e-1;
            lambda = 1e-4;
            srcImg = imgaussfilt(rand(height_,width_),4);
            obsImg = srcImg + wSigma*randn(height_,width_);
            
            % Expected values
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', nVms,... 
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            lppufb.Angles = angs;
            
            % Instantiation of target class
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            itermethod = IstaSystem('Lambda',lambda);
            testCase.target = IterativeSparseApproximater(...
                'Algorithm',itermethod,...
                'Dictionary', {synthesizer,analyzer});
            
            % Actual values
            resActual = testCase.target.step(obsImg);
            
            % Evaluation
            testCase.verifySize(resActual, size(srcImg));
            
            import matlab.unittest.constraints.IsLessThan
            rmse = @(x,y) norm(x(:)-y(:))/sqrt(numel(x));
            rmsePre = rmse(srcImg,obsImg);
            rmsePst = rmse(srcImg,resActual);
            testCase.verifyThat(rmsePst,IsLessThan(rmsePre));
            
        end
        
        function testStepIsta3dDecimation(testCase,...
                vdec, hdec, ddec)
            
            nDecs = [ vdec hdec ddec ];
            nChs  = [ 5 5 ];
            nOrds = [ 2 2 2 ];
            nVms  = 1;
            height_ = 16;
            width_  = 16;
            depth_  = 16;
            wSigma = 1e-1;
            lambda = 1e-4;
            srcImg = imgaussfilt3(rand(height_,width_,depth_),4);
            obsImg = srcImg + wSigma*randn(height_,width_,depth_);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', nVms,...                 
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            lppufb.Angles = angs;
            
            % Instantiation of target class
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            itermethod = IstaSystem('Lambda',lambda);
            testCase.target = IterativeSparseApproximater(...
                'Algorithm',itermethod,...
                'Dictionary', {synthesizer,analyzer});
            
            % Actual values
            resActual = testCase.target.step(obsImg);
            
            % Evaluation
            testCase.verifySize(resActual, size(srcImg));
            
            import matlab.unittest.constraints.IsLessThan
            rmse = @(x,y) norm(x(:)-y(:))/sqrt(numel(x));
            rmsePre = rmse(srcImg,obsImg);
            rmsePst = rmse(srcImg,resActual);
            testCase.verifyThat(rmsePst,IsLessThan(rmsePre));
            
        end
        
        function testStepIsta2dChannel(testCase,...
                sch, ach)
            
            nDecs = [ 2 2 ];
            nChs  = [ sch ach ];
            nOrds = [ 4 4 ];
            nVms = 1;
            height_ = 16;
            width_  = 16;
            wSigma = 1e-1;
            lambda = 1e-4;
            srcImg = imgaussfilt(rand(height_,width_),4);
            obsImg = srcImg + wSigma*randn(height_,width_);
            
            % Expected values
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', nVms,... 
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            lppufb.Angles = angs;
            
            % Instantiation of target class
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            itermethod = IstaSystem('Lambda',lambda);
            testCase.target = IterativeSparseApproximater(...
                'Algorithm',itermethod,...
                'Dictionary', {synthesizer,analyzer});
            
            % Actual values
            resActual = testCase.target.step(obsImg);
            
            % Evaluation
            testCase.verifySize(resActual, size(srcImg));
            
            import matlab.unittest.constraints.IsLessThan
            rmse = @(x,y) norm(x(:)-y(:))/sqrt(numel(x));
            rmsePre = rmse(srcImg,obsImg);
            rmsePst = rmse(srcImg,resActual);
            testCase.verifyThat(rmsePst,IsLessThan(rmsePre));
            
        end
        
        function testStepIsta3dChannel(testCase,...
                sch, ach)
            
            nDecs = [ 2 2 1 ];
            nChs  = [ sch ach ];
            nOrds = [ 2 2 2 ];
            nVms  = 1;
            height_ = 16;
            width_  = 16;
            depth_  = 16;
            wSigma = 1e-1;
            lambda = 1e-4;
            srcImg = imgaussfilt3(rand(height_,width_,depth_),4);
            obsImg = srcImg + wSigma*randn(height_,width_,depth_);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', nVms,...                 
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            lppufb.Angles = angs;
            
            % Instantiation of target class
            import saivdr.sparserep.*
            import saivdr.restoration.ista.*
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            itermethod = IstaSystem('Lambda',lambda);
            testCase.target = IterativeSparseApproximater(...
                'Algorithm',itermethod,...
                'Dictionary', {synthesizer,analyzer});
            
            % Actual values
            resActual = testCase.target.step(obsImg);
            
            % Evaluation
            testCase.verifySize(resActual, size(srcImg));
            
            import matlab.unittest.constraints.IsLessThan
            rmse = @(x,y) norm(x(:)-y(:))/sqrt(numel(x));
            rmsePre = rmse(srcImg,obsImg);
            rmsePst = rmse(srcImg,resActual);
            testCase.verifyThat(rmsePst,IsLessThan(rmsePre));
            
        end        
            
    end
    
end
