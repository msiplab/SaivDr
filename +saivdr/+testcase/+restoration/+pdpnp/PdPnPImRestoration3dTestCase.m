classdef PdPnPImRestoration3dTestCase < matlab.unittest.TestCase
    %PDPNPIMRESTORATION3DTESTCASE Test Case for PdPnPImRestoration3d
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
    % Problem setting:
    %
    %   x* = argmin_x F(x) + \lambda R(x) s.t. Dx \in C
    %
    % - Cost function (Fidelity)
    %   
    %   F(x) = 1/2 ||v-PDx||^2
    %
    % - Regularizer
    %
    %   R(x): Determine explisitly or implisitly. 
    %
    % - Hard constraint
    %
    %   C: Convex set
    %
    % - Measuremnt process
    %
    %   v = Py
    %
    % - Dictionary
    %
    %   y = Dx 
    %
    % =================================================================
    % Algorithm: Primal-dual Plug-and-Play Image Restoration
    % =================================================================    
    % Input:  x(0), y1(0),y2(0)
    % Output: x(n)
    % 1: while A stopping criterin is not satisfied do
    % 2:    x(n+1) = GR( x(n)-\gamma1(D'(P'y1(n)+y2(n))) ,sqrt(\gamma1))
    % 3:    y1(n) <- y1(n) + \gamma2 PD(2x(n+1)-x(n))
    % 4:    y2(n) <- y2(n) + \gamma2 D(2x(n+1)-x(n))
    % 5:    y1(n+1) = y1(n) - \gamma2 prox_{1/\gamma2}F(1/gamma2 y1(n)) 
    % 6:    y2(n+1) = y2(n) - \gamma2 P_C(1/gamma2 y2(n))     
    % 7:    n <- n+1
    % 8: end while
    %    ||
    % Analysis-synthesis expression
    %    ||
    % =================================================================
    % Algorithm: Primal-dual Plug-and-Play Image Restoration
    % =================================================================    
    % Input:  x(0), y1(0),y2(0)
    % Output: x(n)
    % 1: while A stopping criterin is not satisfied do
    % 2:    v <- D'(P'y1(n)+y2(n)) % <-- Analyze
    % 3:    x(n+1) = G_R( x(n)-\gamma1 v ,sqrt(\gamma1))
    % 4:    w <- D(2x(n+1)-x(n))   % <-- Synthesize
    % 5:    y1(n) <- y1(n) + \gamma2 Pw
    % 6:    y2(n) <- y2(n) + \gamma2 w
    % 7:    y1(n+1) = y1(n) - \gamma2 prox_{1/\gamma2}F(1/gamma2 y1(n)) 
    % 8:    y2(n+1) = y2(n) - \gamma2 P_C(1/gamma2 y2(n))     
    % 9:    n <- n+1
    %10: end while
    %    
    
    properties
        pdpnpimrstr
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.pdpnpimrstr);
        end
    end
    
    methods (Test)
        
        function testConstruction(testCase)
        
            % Parameters
            lambda = 0.01;
            gamma1 = 0.01;
            
            % Expectation
            lambdaExpctd = lambda; % Regularization parameter
            gamma1Expctd = gamma1; % Step size
            
            % Instantiation 
            import saivdr.restoration.pdpnp.PdPnPImRestoration3d;
            testCase.pdpnpimrstr = PdPnPImRestoration3d();
            
            % Actual
            lambdaActual = testCase.pdpnpimrstr.Lambda; 
            gamma1Actual = testCase.pdpnpimrstr.Gamma1;  
            
            % Evaluation
            testCase.verifyEqual(lambdaActual,lambdaExpctd);            
            testCase.verifyEqual(gamma1Actual,gamma1Expctd);
            
        end        
        
        %{
        % Test denoising
        function testPdsNsoltDeNoise(testCase)
            
            % Preperation
            height = 32;
            width  = 32;
            depth  = 16;
            srcImg = rand(height,width,depth);
            
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem();
            analyzer = NsoltFactory.createAnalysis3dSystem();
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = BlurSystem(...
                'DataType','Volumetric Data');
            noiseProcess  = AdditiveWhiteGaussianNoiseSystem();
            degradation  = DegradationSystem(...
                'LinearProcess',linearProcess,...
                'NoiseProcess',noiseProcess);
            obsImg = step(degradation,srcImg);
            
            % MSE before processing
            mse = @(x,y) sum((x(:)-y(:)).^2)/numel(x);
            msePre = mse(obsImg,srcImg);
            
            % Instantiation of target class
            import saivdr.restoration.pds.*
            testCase.pdsimrstr = PdsImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'NumberOfTreeLevels',3,...
                'Lambda',1e-2);
            
            % MSE after processing
            resImg = step(testCase.pdsimrstr,obsImg);
            
            % Evaluation
            msePst = mse(resImg,srcImg);
            
            import matlab.unittest.constraints.IsLessThan
            testCase.assertThat(msePst,IsLessThan(msePre));
            
            %             figure(1), imshow(srcImg)
            %             figure(2), imshow(obsImg)
            %             figure(3), imshow(resImg)
            
        end
     
        % Test denoising
        function testPdsNsoltDeBlur(testCase)
            
            % Preperation
            srcImg = rand(32,32,16);
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem();
            analyzer = NsoltFactory.createAnalysis3dSystem();
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = BlurSystem(...
                'DataType','Volumetric Data',...
                'BlurType','Gaussian');
            noiseProcess  = AdditiveWhiteGaussianNoiseSystem();
            degradation  = DegradationSystem(...
                'LinearProcess',linearProcess,...
                'NoiseProcess',noiseProcess);
            obsImg = step(degradation,srcImg);
            
            % MSE before processing
            mse = @(x,y) sum((x(:)-y(:)).^2)/numel(x);
            msePre = mse(obsImg,srcImg);
            
            % Instantiation of target class
            import saivdr.restoration.pds.*
            testCase.pdsimrstr = PdsImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'NumberOfTreeLevels',3,...
                'Lambda',1e-2);
            
            % MSE after processing
            resImg = step(testCase.pdsimrstr,obsImg);
            
            % Evaluation
            msePst = mse(resImg,srcImg);
            
            import matlab.unittest.constraints.IsLessThan
            testCase.assertThat(msePst,IsLessThan(msePre));
            
            %              figure(1), imshow(srcImg)
            %              figure(2), imshow(obsImg)
            %              figure(3), imshow(resImg)
            
        end
        
        % Test inpainting
        function testPdsNsoltInPainting(testCase)
            
            % Preperation
            srcImg = rand(32,32,16);
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem();
            analyzer = NsoltFactory.createAnalysis3dSystem();
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = PixelLossSystem(...
                'DataType','Volumetric Data');
            noiseProcess = NoiselessSystem();
            degradation  = DegradationSystem(...
                'LinearProcess',linearProcess,...
                'NoiseProcess',noiseProcess);
            obsImg = step(degradation,srcImg);
            
            % MSE before processing
            mse = @(x,y) sum((x(:)-y(:)).^2)/numel(x);
            msePre = mse(obsImg,srcImg);
            
            % Instantiation of target class
            import saivdr.restoration.pds.*
            testCase.pdsimrstr = PdsImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'NumberOfTreeLevels',3,...
                'Lambda',1e-2);
            
            % MSE after processing
            resImg = step(testCase.pdsimrstr,obsImg);
            
            % Evaluation
            msePst = mse(resImg,srcImg);
            
            import matlab.unittest.constraints.IsLessThan
            testCase.assertThat(msePst,IsLessThan(msePre));
            
            %               figure(1), imshow(srcImg)
            %               figure(2), imshow(obsImg)
            %               figure(3), imshow(resImg)
            
        end
        
        % Test super resolution
        function testPdsNsoltSuperResolution(testCase)
            
            % Preperation
            srcImg = rand(32,32,16);
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem();
            analyzer = NsoltFactory.createAnalysis3dSystem();
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = DecimationSystem(...
                'DataType','Volumetric Data');
            noiseProcess = AdditiveWhiteGaussianNoiseSystem();
            degradation  = DegradationSystem(...
                'LinearProcess',linearProcess,...
                'NoiseProcess',noiseProcess);
            obsImg = step(degradation,srcImg);
            
            % MSE before processing
            mse = @(x,y) sum((x(:)-y(:)).^2)/numel(x);
            resImg = ...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(obsImg,...
                2),1),...
                2),1),...
                2),1);
            kernel = ones([2 2 2])/8;
            resImg = imfilter(resImg,kernel);
            msePre = mse(resImg,srcImg);
            
            % Instantiation of target class
            import saivdr.restoration.pds.*
            testCase.pdsimrstr = PdsImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'NumberOfTreeLevels',3,...
                'Lambda',1e-2);
            
            % MSE after processing
            resImg = step(testCase.pdsimrstr,obsImg);
            
            % Evaluation
            msePst = mse(resImg,srcImg);
            
            
            import matlab.unittest.constraints.IsLessThan
            testCase.assertThat(msePst,IsLessThan(msePre));
            
            %                figure(1), imshow(srcImg)
            %                figure(2), imshow(obsImg)
            %                figure(3), imshow(resImg)
            
        end
        
        % Test step monitoring
        function testStepMonitoring(testCase)
            
            % Preperation
            srcImg = rand(32,32,16);
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem();
            analyzer = NsoltFactory.createAnalysis3dSystem();
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = BlurSystem(...
                'DataType','Volumetric Data');
            noiseProcess  = AdditiveWhiteGaussianNoiseSystem();
            import saivdr.utility.*
            stepMonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'DataType','Volumetric Data',...
                'IsMSE',true);
            degradation  = DegradationSystem(...
                'LinearProcess',linearProcess,...
                'NoiseProcess',noiseProcess);
            obsImg = step(degradation,srcImg);
            
            % Definition of MSE
            mse = @(x,y) sum((double(x(:))-double(y(:))).^2)/numel(x);
            
            % Instantiation of target class
            import saivdr.restoration.pds.*
            testCase.pdsimrstr = PdsImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'NumberOfTreeLevels',3,...
                'Lambda',1e-2,...
                'StepMonitor',stepMonitor);
            
            % MSE after processing
            resImg = step(testCase.pdsimrstr,obsImg);
            mseExpctd = mse(uint8(255*resImg),uint8(255*srcImg));
            
            % Actual value
            mses = get(stepMonitor,'MSEs');
            nitr = get(stepMonitor,'nItr');
            mseActual = mses(nitr);
            
            % Evaluation
            testCase.assertEqual(mseActual,mseExpctd)
            
        end
        
        % Test denoising
        function testPdsUdHaarDeNoise(testCase)
            
            % Preperation
            srcImg = rand(32,32,64);
            import saivdr.dictionary.udhaar.*
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer = UdHaarAnalysis3dSystem();
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = BlurSystem(...
                'DataType','Volumetric Data');
            noiseProcess  = AdditiveWhiteGaussianNoiseSystem();
            degradation  = DegradationSystem(...
                'LinearProcess',linearProcess,...
                'NoiseProcess',noiseProcess);
            obsImg = step(degradation,srcImg);
            
            % MSE before processing
            mse = @(x,y) sum((x(:)-y(:)).^2)/numel(x);
            msePre = mse(obsImg,srcImg);
            
            % Instantiation of target class
            import saivdr.restoration.pds.*
            testCase.pdsimrstr = PdsImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'NumberOfTreeLevels',3,...
                'Lambda',1e-2);
            
            % MSE after processing
            resImg = step(testCase.pdsimrstr,obsImg);
            
            % Evaluation
            msePst = mse(resImg,srcImg);
            
            import matlab.unittest.constraints.IsLessThan
            testCase.assertThat(msePst,IsLessThan(msePre));
            
            %             figure(1), imshow(srcImg)
            %             figure(2), imshow(obsImg)
            %             figure(3), imshow(resImg)
            
        end
        
        % Test denoising
        function testAbsetProperties(testCase)
            
            % Preperation
            import saivdr.dictionary.udhaar.*
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer = UdHaarAnalysis3dSystem();
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = BlurSystem(...
                'DataType','Volumetric Data');
            
            % Abset synthesizer
            try
                import saivdr.restoration.pds.*
                testCase.pdsimrstr = PdsImRestoration3d(...
                    'AdjOfSynthesizer',analyzer,...
                    'LinearProcess',linearProcess);
            catch
            end
            
            % Abset ajoint of synthesizer
            try
                testCase.pdsimrstr = PdsImRestoration3d(...
                    'Synthesizer',synthesizer,...
                    'LinearProcess',linearProcess);
            catch
            end
            
            % Abset ajoint of synthesizer
            try
                testCase.pdsimrstr = PdsImRestoration3d(...
                    'AdjOfSynthesizer',analyzer,...
                    'Synthesizer',synthesizer,...
                    'LinearProcess',linearProcess);
            catch
            end
            
        end

        % Test
        function testAbsentProperties(testCase)
            
            % Preperation
            import saivdr.dictionary.udhaar.*
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer = UdHaarAnalysis3dSystem();
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = BlurSystem(...
                'DataType','Volumetric Data');
            
            % Expected value
            exceptionIdExpctd = 'SaivDr:InstantiationException';
            import saivdr.restoration.pds.*
            
            
            % Abset synthesizer
            messageExpctd = 'Synthesizer must be given.';
            try
                testCase.pdsimrstr = PdsImRestoration3d(...
                    'AdjOfSynthesizer',analyzer,...
                    'LinearProcess',   linearProcess);
                step(testCase.pdsimrstr,[]);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
            
            % Abset ajoint of synthesizer
            messageExpctd = 'AdjOfSynthesizer must be given.';
            try
                testCase.pdsimrstr = PdsImRestoration3d(...
                    'Synthesizer',   synthesizer,...
                    'LinearProcess', linearProcess);
                step(testCase.pdsimrstr,[]);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
            
            % Abset ajoint of synthesizer
            messageExpctd = 'LinearProcess must be given.';
            try
                testCase.pdsimrstr = PdsImRestoration3d(...
                    'AdjOfSynthesizer',analyzer,...
                    'Synthesizer',   synthesizer);
                step(testCase.pdsimrstr,[]);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
        end
        
        % Test
        function testCloneWithUdHaar(testCase)
            
            % Preperation
            srcImg = checkerboard(8,4,4);
            import saivdr.dictionary.udhaar.*
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer = UdHaarAnalysis3dSystem();
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = BlurSystem(...
                'DataType','Volumetric Data');
            noiseProcess  = AdditiveWhiteGaussianNoiseSystem();
            degradation  = DegradationSystem(...
                'LinearProcess',linearProcess,...
                'NoiseProcess',noiseProcess);
            obsImg = step(degradation,srcImg);
            
            % MSE before processing
            mse = @(x,y) sum((x(:)-y(:)).^2)/numel(x);
            
            % Instantiation of target class
            import saivdr.restoration.pds.*
            testCase.pdsimrstr = PdsImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'NumberOfTreeLevels',3,...
                'Lambda',1e-2);
            
            % MSE by original object
            resImg = step(testCase.pdsimrstr,obsImg);
            mseOrg = mse(resImg,srcImg);
            
            % Instantiation of target class
            clonepdsimrstr = clone(testCase.pdsimrstr);
            
            % MSE by clone object
            resImg = step(clonepdsimrstr,obsImg);
            mseCln = mse(resImg,srcImg);
            
            % Evaluation
            testCase.verifyEqual(mseCln,mseOrg);
            
        end
        
        % Test
        function testCloneWithNsolt(testCase)
            
            % Preperation
            srcImg = rand(32,32,16);
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem();
            analyzer = NsoltFactory.createAnalysis3dSystem();
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = BlurSystem(...
                'DataType','Volumetric Data');
            noiseProcess  = AdditiveWhiteGaussianNoiseSystem();
            degradation  = DegradationSystem(...
                'LinearProcess',linearProcess,...
                'NoiseProcess',noiseProcess);
            obsImg = step(degradation,srcImg);
            
            % MSE before processing
            mse = @(x,y) sum((x(:)-y(:)).^2)/numel(x);
            
            % Instantiation of target class
            import saivdr.restoration.pds.*
            testCase.pdsimrstr = PdsImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'NumberOfTreeLevels',3,...
                'Lambda',1e-2);
            
            % MSE by original object
            resImg = step(testCase.pdsimrstr,obsImg);
            mseOrg = mse(resImg,srcImg);
            
            % Instantiation of target class
            clonepdsimrstr = clone(testCase.pdsimrstr);
            
            % MSE by clone object
            resImg = step(clonepdsimrstr,obsImg);
            mseCln = mse(resImg,srcImg);
            
            % Evaluation
            testCase.verifyEqual(mseCln,mseOrg);
            
        end
        
        % Test
        function testCloneWithNsoltChildObj(testCase)
            
            % Preperation
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem();
            analyzer = NsoltFactory.createAnalysis3dSystem();
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = BlurSystem(...
                'DataType','Volumetric Data');
            
            % Instantiation of target class
            import saivdr.restoration.pds.*
            testCase.pdsimrstr = PdsImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'NumberOfTreeLevels',3,...
                'Lambda',1e-2);
            
            % Instantiation of target class
            clonepdsimrstr = clone(testCase.pdsimrstr);
            
            % Evaluation
            testCase.verifyEqual(clonepdsimrstr,testCase.pdsimrstr);
            testCase.verifyFalse(clonepdsimrstr == testCase.pdsimrstr);
            %
            prpOrg = get(testCase.pdsimrstr,'Synthesizer');
            prpCln = get(clonepdsimrstr,'Synthesizer');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            prpOrg = get(testCase.pdsimrstr,'AdjOfSynthesizer');
            prpCln = get(clonepdsimrstr,'AdjOfSynthesizer');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            prpOrg = get(testCase.pdsimrstr,'LinearProcess');
            prpCln = get(clonepdsimrstr,'LinearProcess');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            
        end
        
        % Test
        function testCloneWithUdHaarChildObj(testCase)
            
            % Preperation
            import saivdr.dictionary.udhaar.*
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer = UdHaarAnalysis3dSystem();
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = BlurSystem(...
                'DataType','Volumetric Data');
            
            % Instantiation of target class
            import saivdr.restoration.pds.*
            testCase.pdsimrstr = PdsImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'NumberOfTreeLevels',3,...
                'Lambda',1e-2);
            
            % Instantiation of target class
            clonepdsimrstr = clone(testCase.pdsimrstr);
            
            % Evaluation
            testCase.verifyEqual(clonepdsimrstr,testCase.pdsimrstr);
            testCase.verifyFalse(clonepdsimrstr == testCase.pdsimrstr);
            %
            prpOrg = get(testCase.pdsimrstr,'Synthesizer');
            prpCln = get(clonepdsimrstr,'Synthesizer');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            prpOrg = get(testCase.pdsimrstr,'AdjOfSynthesizer');
            prpCln = get(clonepdsimrstr,'AdjOfSynthesizer');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            prpOrg = get(testCase.pdsimrstr,'LinearProcess');
            prpCln = get(clonepdsimrstr,'LinearProcess');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            
        end
        %}
    end
    
end

