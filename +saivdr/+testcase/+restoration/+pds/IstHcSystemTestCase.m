classdef IstHcSystemTestCase < matlab.unittest.TestCase
    %ISTHCSYSTEMTESTCASE Test case for IstHcSystem
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
        useparallel = struct('true', true, 'false', false );
        usegpu = struct('true', true, 'false', false );
        niter = struct('small',1, 'large', 16);
        depth = struct('small',8, 'large', 16);
        hight = struct('small',8, 'large', 16);
        width = struct('small',8, 'large', 16);
        dsplit = struct('small',1, 'large', 2);
        vsplit = struct('small',1, 'large', 2);
        hsplit = struct('small',1, 'large', 2);
        nlevels = struct('small', 1, 'large', 3 );
        vmin = struct('neginf', -Inf,'negone', -1,'zero', 0 )
        vmax = struct('posinf', Inf, 'posone', 1)
    end
    
    properties
        target
    end
    
    methods (TestMethodTeardown)
        function deteleObject(testCase)
            delete(testCase.target);
        end
    end
    
    methods (Test)
        
        function testConstruction(testCase)
            
            % Parameters
            import saivdr.degradation.linearprocess.IdenticalMappingSystem
            lambdaExpctd  = 0; % Regularization parameter
            gammaExpctd   = []; % Step size parameter
            msrExpctd = IdenticalMappingSystem;
            mpjExpctd = [];
            dicExpctd = [];
            gdnExpctd = [];
            obsExpctd = [];
            
            % Instantiation
            import saivdr.restoration.pds.*
            testCase.target = IstHcSystem();
            
            % Actual values
            lambdaActual  = testCase.target.Lambda;  % Regularizaton parameter
            gammaActual  = testCase.target.Gamma;  % Stepsize parameter
            msrActual = testCase.target.MeasureProcess;
            mpjActual = testCase.target.MetricProjection;
            dicActual = testCase.target.Dictionary;
            gdnActual = testCase.target.GaussianDenoiser;
            obsActual = testCase.target.Observation;
            
            % Evaluation
            testCase.verifyEqual(lambdaActual,lambdaExpctd);
            testCase.verifyEqual(gammaActual,gammaExpctd);
            testCase.verifyEqual(msrActual,msrExpctd);
            testCase.verifyEqual(mpjActual,mpjExpctd);            
            testCase.verifyEqual(dicActual,dicExpctd);
            testCase.verifyEqual(gdnActual,gdnExpctd);
            testCase.verifyEqual(obsActual,obsExpctd);
            
        end

        function testStepVolumetricDataBoxConst(testCase,...
                niter,depth,width,vmin,vmax,nlevels)
            
            % Parameters
            lambda = 1e-3;
            phtm = phantom('Modified Shepp-Logan',depth);
            sliceYZ = permute(phtm,[1 3 2]);
            uSrc = repmat(sliceYZ,[1 width 1]);
            
            % Instantiation of observation
            import saivdr.degradation.linearprocess.*
            pSigma = 2.00; % Extent of PSF
            wSigma = 1e-3; % Standard deviation of noise
            msrProc = BlurSystem(...
                'BlurType','Gaussian',...
                'SigmaOfGaussianKernel',pSigma,...
                'ProcessingMode','Forward');
            import saivdr.restoration.metricproj.*
            mtrProj = ProxBoxConstraint(...
                'Vmin', vmin,...
                'Vmax', vmax);
            vObs = msrProc.step(uSrc) ...
                + wSigma*randn(size(uSrc));
            
            % Instantiation of dictionary
            import saivdr.dictionary.udhaar.*
            fwdDic  = UdHaarSynthesis3dSystem();
            adjDic  = UdHaarAnalysis3dSystem('NumberOfLevels',nlevels);
            
            % Calculation of step size parameter
            framebound = fwdDic.FrameBound;
            step(msrProc,vObs);

            % Expected values
            % gamma1*gamma2*(sigma_MAX(L))^2 < 1
            gammaExpctd{1} = 1/(framebound*msrProc.LambdaMax);
            gammaExpctd{2} = 1/(1.05*gammaExpctd{1}*framebound);            
            lambdaExpctd = lambda;
            thr = lambdaExpctd*gammaExpctd{1};
            softthresh = @(x) sign(x).*max(abs(x)-thr,0);
            [x0,scale] = adjDic.step(zeros(size(vObs),'like',vObs));
            adjProc = msrProc.clone();
            adjProc.release();
            adjProc.ProcessingMode = 'Adjoint';
            resExpctd = cell(niter,1);
            xPre = x0;
            yPre = zeros(size(vObs));
            % Steps
            resPre = fwdDic(x0,scale);
            for iter = 1:niter
                t = adjDic.step(adjProc.step(msrProc.step(resPre)-vObs)+yPre);
                x = softthresh(xPre-gammaExpctd{1}*t);
                u = 2*fwdDic.step(x,scale) - resPre;
                y = yPre + gammaExpctd{2}*u;
                pcy = y/gammaExpctd{2};
                % 
                pcy(pcy<=vmin) = vmin;
                pcy(pcy>=vmax) = vmax;
                %
                y = y - gammaExpctd{2}*pcy;
                resExpctd{iter} = (u+resPre)/2;
                resPre = resExpctd{iter};
                xPre = x;
                yPre = y;
            end
            coefsExpctd = x;
            scalesExpctd = scale;
            
            % Instantiation of test target
            import saivdr.restoration.pds.*
            testCase.target = IstHcSystem(...
                'Observation',    vObs,...
                'DataType', 'Volumetric Data',...
                'Lambda',         lambda,...
                'MeasureProcess', msrProc,...
                'MetricProjection', mtrProj,...
                'Dictionary', { fwdDic, adjDic } );
            
            % Evaluation of 1st step
            eps = 1e-10;
            iterExpctd = 1;
            [resActual,rmseActual] = testCase.target.step();
            iterActual  = testCase.target.Iteration;
            gammaActual = testCase.target.Gamma;
            lambdaActual = testCase.target.Lambda;
            %
            testCase.verifyEqual(iterActual,iterExpctd)
            testCase.verifyEqual(gammaActual,gammaExpctd)
            testCase.verifyEqual(lambdaActual,lambdaExpctd)
            %
            testCase.verifySize(resActual,size(uSrc));
            %
            diff = max(abs(resExpctd{iterExpctd}(:)-resActual(:)));
            testCase.verifyEqual(resActual,resExpctd{iterExpctd},...
                'AbsTol',eps,num2str(diff));
            %
            % Evaluation of iterative step
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            import matlab.unittest.constraints.IsGreaterThanOrEqualTo            
            rmse = @(x,y) norm(x(:)-y(:),2)/sqrt(numel(x));
            rmsePre = rmseActual;
            resPre  = resActual;
            for iter = 2:niter
                iterExpctd = iterExpctd + 1;
                %
                [resActual,rmseActual] = testCase.target.step();
                iterActual = testCase.target.Iteration;
                %
                testCase.verifyEqual(iterActual,iterExpctd)
                diff = max(abs(resExpctd{iterExpctd}(:)-resActual(:)));
                testCase.verifyEqual(resActual,resExpctd{iterExpctd},...
                    'AbsTol',eps,num2str(diff));
                %
                rmseExpctd = rmse(resActual,resPre);
                diff = max(abs(rmseExpctd-rmseActual));
                testCase.verifyEqual(rmseActual,rmseExpctd,...
                    'AbsTol',eps,num2str(diff));
                %
                testCase.verifyThat(rmseActual,IsLessThanOrEqualTo(rmsePre))
                %
                resPre  = resActual;
            end

            % Evaluate hard constraint
            if niter > 1
                tol = 1e-2;
                vminActual = min(resActual(:));
                vmaxActual = max(resActual(:));
                testCase.verifyThat(vminActual+tol,IsGreaterThanOrEqualTo(vmin));
                testCase.verifyThat(vmaxActual-tol,IsLessThanOrEqualTo(vmax));
            end
            
            % Actual values of coefficients
            [coefsActual,scalesActual] = testCase.target.getCoefficients();
            
            % Evaluation of coefficients
            testCase.verifySize(scalesActual,size(scalesExpctd));
            diff = max(abs(scalesExpctd(:) - scalesActual(:)));
            testCase.verifyEqual(scalesActual,scalesExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            %
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));                        

        end
        

        function testStepSplit(testCase,...
                depth,dsplit,useparallel,usegpu)
            
            % Parameters
            vmin_ = -1;
            vmax_ = 1;
            niter_ = 4;
            nlevels_ = 3;
            splitfactor = [2*ones(1,2) dsplit];
            padsize = 2^(nlevels_-1)*ones(1,3);            
            lambda = 1e-3;
            width_ = 8;
            phtm = phantom('Modified Shepp-Logan',depth);
            sliceYZ = permute(phtm,[1 3 2]);
            uSrc = repmat(sliceYZ,[1 width_ 1]);
            
            % Instantiation of observation
            import saivdr.degradation.linearprocess.*
            pSigma = 2.00; % Extent of PSF
            wSigma = 1e-3; % Standard deviation of noise
            msrProc = BlurSystem(...
                'BlurType','Gaussian',...
                'SigmaOfGaussianKernel',pSigma,...
                'ProcessingMode','Forward');
            import saivdr.restoration.metricproj.*
            mtrProj = ProxBoxConstraint(...
                'Vmin', vmin_,...
                'Vmax', vmax_);
            vObs = msrProc.step(uSrc) ...
                + wSigma*randn(size(uSrc));
            
            % Instantiation of dictionary
            import saivdr.dictionary.udhaar.*
            fwdDic  = UdHaarSynthesis3dSystem();
            adjDic  = UdHaarAnalysis3dSystem('NumberOfLevels',nlevels_);
            
            % Instantiation of reference
            import saivdr.restoration.pds.*
            reference = IstHcSystem(...
                'Observation',    vObs,...
                'DataType', 'Volumetric Data',...
                'Lambda',         lambda,...
                'MeasureProcess', msrProc,...
                'MetricProjection', mtrProj,...
                'Dictionary', { fwdDic, adjDic } );
            
            testCase.target = IstHcSystem(...
                'Observation',    vObs,...
                'DataType', 'Volumetric Data',...
                'Lambda',         lambda,...
                'MeasureProcess', msrProc,...
                'MetricProjection', mtrProj,...
                'Dictionary', { fwdDic, adjDic },...
                'SplitFactor', splitfactor,...
                'PadSize', padsize,...
                'UseParallel', useparallel,...
                'UseGpu', usegpu);
            
            % Restoration
            for iter = 1:niter_
                resExpctd = reference.step();
                resActual = testCase.target.step();
            end
            
            % Evaluation of result
            eps = 1e-10;
            %import matlab.unittest.constraints.IsLessThan
            testCase.verifySize(resActual,size(resExpctd));
            diff = max(abs(resExpctd(:) - resActual(:)));
            testCase.verifyEqual(resActual,resExpctd,...
                'AbsTol',eps,sprintf('%g',diff));
            
            % Evaluation of coefficients
            [coefsExpctd,scalesExpctd] = reference.getCoefficients();
            [coefsActual,scalesActual] = testCase.target.getCoefficients();
            %
            testCase.verifySize(scalesActual,size(scalesExpctd));
            diff = max(abs(scalesExpctd(:) - scalesActual(:)));
            testCase.verifyEqual(scalesActual,scalesExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            %
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));                        
            
        end
        
        function testIsLambdaCompensation(testCase,...
            depth,width,nlevels,useparallel)
            
            % Parameters
            lambda = 1e-3;
            islambdacomp = true;
            %
            vmin_ = 0;
            vmax_ = 1;
            usegpu_ = false;
            dsplit_ = 2;
            splitfactor = [2*ones(1,2) dsplit_];
            padsize = 2^(nlevels-1)*ones(1,3);
            phtm = phantom('Modified Shepp-Logan',depth);
            sliceYZ = permute(phtm,[1 3 2]);
            uSrc = 0.5*repmat(sliceYZ,[1 width 1]) + 1;
            
            % Instantiation of observation
            import saivdr.degradation.linearprocess.*
            import saivdr.restoration.metricproj.*
            pSigma = 2.00; % Extent of PSF
            wSigma = 1e-3; % Standard deviation of noise
            msrProc = BlurSystem(...
                'BlurType','Gaussian',...
                'SigmaOfGaussianKernel',pSigma,...
                'ProcessingMode','Forward');
            mtrProj = ProxBoxConstraint(...
                'Vmin', vmin_,...
                'Vmax', vmax_);            
            vObs = msrProc.step(uSrc) ...
                + wSigma*randn(size(uSrc));
            
            % Instantiation of dictionary
            import saivdr.dictionary.udhaar.*
            fwdDic  = UdHaarSynthesis3dSystem();
            adjDic  = UdHaarAnalysis3dSystem('NumberOfLevels',nlevels);
            coefs = adjDic.step(vObs);
            
            % Instantiation of reference
            import saivdr.restoration.pds.*
            testCase.target = IstHcSystem(...
                'Observation',    vObs,...
                'DataType', 'Volumetric Data',...
                'Lambda',         lambda,...
                'MeasureProcess', msrProc,...
                'MetricProjection', mtrProj,...                
                'Dictionary', { fwdDic, adjDic } ,...
                'SplitFactor', splitfactor,...
                'PadSize', padsize,...
                'UseParallel', useparallel,...
                'UseGpu', usegpu_,...
                'IsLambdaCompensation',islambdacomp);
            
            % Expected value
            lambdaExpctd = lambda * numel(vObs)^2/numel(coefs);
            
            % Actual value
            testCase.target.step();
            lambdaActual = testCase.target.Lambda;
            
            % Evaluation
            eps = 1e-10;
            %import matlab.unittest.constraints.IsLessThan
            diff = max(abs(lambdaExpctd - lambdaActual));
            testCase.verifyEqual(lambdaActual,lambdaExpctd,...
                'AbsTol',eps,sprintf('%g',diff));
        end

        
        
        function testStepVolumetricDataNormBallConst(testCase,...
                niter,depth,width,nlevels)
            
            % Parameters
            lambda = 1e-3;
            phtm = phantom('Modified Shepp-Logan',depth);
            sliceYZ = permute(phtm,[1 3 2]);
            uSrc = repmat(sliceYZ,[1 width 1]);
            epsy   = 0.5*sqrt(numel(uSrc));
            center = 0.5*ones(size(uSrc));
            
            % Instantiation of observation
            import saivdr.degradation.linearprocess.*
            pSigma = 2.00; % Extent of PSF
            wSigma = 1e-3; % Standard deviation of noise
            msrProc = BlurSystem(...
                'BlurType','Gaussian',...
                'SigmaOfGaussianKernel',pSigma,...
                'ProcessingMode','Forward');
            import saivdr.restoration.metricproj.*
            mtrProj = ProxNormBallConstraint(...
                'Eps', epsy,...
                'Center', center);
            vObs = msrProc.step(uSrc) ...
                + wSigma*randn(size(uSrc));
            
            % Instantiation of dictionary
            import saivdr.dictionary.udhaar.*
            fwdDic  = UdHaarSynthesis3dSystem();
            adjDic  = UdHaarAnalysis3dSystem('NumberOfLevels',nlevels);
            
            % Calculation of step size parameter
            framebound = fwdDic.FrameBound;
            step(msrProc,vObs);

            % Expected values
            % gamma1*gamma2*(sigma_MAX(L))^2 < 1
            gammaExpctd{1} = 1/(framebound*msrProc.LambdaMax);
            gammaExpctd{2} = 1/(1.05*gammaExpctd{1}*framebound);            
            lambdaExpctd = lambda;
            thr = lambdaExpctd*gammaExpctd{1};
            softthresh = @(x) sign(x).*max(abs(x)-thr,0);
            [x0,scale] = adjDic.step(zeros(size(vObs),'like',vObs));
            adjProc = msrProc.clone();
            adjProc.release();
            adjProc.ProcessingMode = 'Adjoint';
            resExpctd = cell(niter,1);
            xPre = x0;
            yPre = zeros(size(vObs));
            % Steps
            resPre = fwdDic(x0,scale);
            for iter = 1:niter
                t = adjDic.step(adjProc.step(msrProc.step(resPre)-vObs)+yPre);
                x = softthresh(xPre-gammaExpctd{1}*t);
                u = 2*fwdDic.step(x,scale) - resPre;
                y = yPre + gammaExpctd{2}*u;
                pcy = y/gammaExpctd{2};
                %
                d = norm(pcy(:)-center(:),2);
                if d>epsy
                    pcy = center + (epsy/d)*(pcy-center);
                end
                %
                y = y - gammaExpctd{2}*pcy;
                resExpctd{iter} = (u+resPre)/2;
                resPre = resExpctd{iter};
                xPre = x;
                yPre = y;
            end
            
            % Instantiation of test target
            import saivdr.restoration.pds.*
            testCase.target = IstHcSystem(...
                'Observation',    vObs,...
                'DataType', 'Volumetric Data',...
                'Lambda',         lambda,...
                'MeasureProcess', msrProc,...
                'MetricProjection', mtrProj,...
                'Dictionary', { fwdDic, adjDic } );
            
            % Evaluation of 1st step
            eps = 1e-10;
            iterExpctd = 1;
            [resActual,rmseActual] = testCase.target.step();
            iterActual  = testCase.target.Iteration;
            gammaActual = testCase.target.Gamma;
            lambdaActual = testCase.target.Lambda;
            %
            testCase.verifyEqual(iterActual,iterExpctd)
            testCase.verifyEqual(gammaActual,gammaExpctd)
            testCase.verifyEqual(lambdaActual,lambdaExpctd)
            %
            testCase.verifySize(resActual,size(uSrc));
            %
            diff = max(abs(resExpctd{iterExpctd}(:)-resActual(:)));
            testCase.verifyEqual(resActual,resExpctd{iterExpctd},...
                'AbsTol',eps,num2str(diff));
            %
            % Evaluation of iterative step
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            import matlab.unittest.constraints.IsGreaterThanOrEqualTo            
            rmse = @(x,y) norm(x(:)-y(:),2)/sqrt(numel(x));
            rmsePre = rmseActual;
            resPre  = resActual;
            for iter = 2:niter
                iterExpctd = iterExpctd + 1;
                %
                [resActual,rmseActual] = testCase.target.step();
                iterActual = testCase.target.Iteration;
                %
                testCase.verifyEqual(iterActual,iterExpctd)
                diff = max(abs(resExpctd{iterExpctd}(:)-resActual(:)));
                testCase.verifyEqual(resActual,resExpctd{iterExpctd},...
                    'AbsTol',eps,num2str(diff));
                %
                rmseExpctd = rmse(resActual,resPre);
                diff = max(abs(rmseExpctd-rmseActual));
                testCase.verifyEqual(rmseActual,rmseExpctd,...
                    'AbsTol',eps,num2str(diff));
                %
                testCase.verifyThat(rmseActual,IsLessThanOrEqualTo(rmsePre))
                %
                resPre  = resActual;
            end

            % Evaluate hard constraint
            if niter > 1
                tol = 1e-2;
                distActual = norm(resActual(:)-center(:),2);
                testCase.verifyThat(distActual-tol,IsLessThanOrEqualTo(epsy));
            end
            
        end
        
    end
    
end