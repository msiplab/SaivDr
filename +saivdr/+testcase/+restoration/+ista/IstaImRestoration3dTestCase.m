classdef IstaImRestoration3dTestCase < matlab.unittest.TestCase
    %ISTAIMRESTORATION3DTESTCASE Test Case for IstaImRestoration3d
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2017-2020, Shogo MURAMATSU
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
        istaimrstr
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.istaimrstr);
        end
    end
    
    methods (Test)
        
        % Test denoising
        function testIstaNsoltDeNoise(testCase)
            
            % Preperation
            height = 32;
            width  = 32;
            depth  = 16;
            srcImg = rand(height,width,depth);
            
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem();
            analyzer = NsoltFactory.createAnalysis3dSystem();
            analyzer.NumberOfLevels = 3;            
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
            import saivdr.restoration.ista.*
            testCase.istaimrstr = IstaImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'Lambda',1e-2);
            
            % MSE after processing
            resImg = step(testCase.istaimrstr,obsImg);
            
            % Evaluation
            msePst = mse(resImg,srcImg);
            
            import matlab.unittest.constraints.IsLessThan
            testCase.assertThat(msePst,IsLessThan(msePre));
            
            %             figure(1), imshow(srcImg)
            %             figure(2), imshow(obsImg)
            %             figure(3), imshow(resImg)
            
        end
     
        % Test denoising
        function testIstaNsoltDeBlur(testCase)
            
            % Preperation
            srcImg = rand(32,32,16);
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem();
            analyzer = NsoltFactory.createAnalysis3dSystem();
            analyzer.NumberOfLevels = 3;            
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
            import saivdr.restoration.ista.*
            testCase.istaimrstr = IstaImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'Lambda',1e-2);
            
            % MSE after processing
            resImg = step(testCase.istaimrstr,obsImg);
            
            % Evaluation
            msePst = mse(resImg,srcImg);
            
            import matlab.unittest.constraints.IsLessThan
            testCase.assertThat(msePst,IsLessThan(msePre));
            
            %              figure(1), imshow(srcImg)
            %              figure(2), imshow(obsImg)
            %              figure(3), imshow(resImg)
            
        end
        
        % Test inpainting
        function testIstaNsoltInPainting(testCase)
            
            % Preperation
            srcImg = rand(32,32,16);
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem();
            analyzer = NsoltFactory.createAnalysis3dSystem();
            analyzer.NumberOfLevels = 3;            
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
            import saivdr.restoration.ista.*
            testCase.istaimrstr = IstaImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'Lambda',1e-2);
            
            % MSE after processing
            resImg = step(testCase.istaimrstr,obsImg);
            
            % Evaluation
            msePst = mse(resImg,srcImg);
            
            import matlab.unittest.constraints.IsLessThan
            testCase.assertThat(msePst,IsLessThan(msePre));
            
            %               figure(1), imshow(srcImg)
            %               figure(2), imshow(obsImg)
            %               figure(3), imshow(resImg)
            
        end
        
        % Test super resolution
        function testIstaNsoltSuperResolution(testCase)
            
            % Preperation
            srcImg = rand(32,32,16);
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis3dSystem();
            analyzer = NsoltFactory.createAnalysis3dSystem();
            analyzer.NumberOfLevels = 3;            
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
            import saivdr.restoration.ista.*
            testCase.istaimrstr = IstaImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'Lambda',1e-2);
            
            % MSE after processing
            resImg = step(testCase.istaimrstr,obsImg);
            
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
            analyzer.NumberOfLevels = 3;            
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
            import saivdr.restoration.ista.*
            testCase.istaimrstr = IstaImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'Lambda',1e-2,...
                'StepMonitor',stepMonitor);
            
            % MSE after processing
            resImg = step(testCase.istaimrstr,obsImg);
            mseExpctd = mse(uint8(255*resImg),uint8(255*srcImg));
            
            % Actual value
            mses = get(stepMonitor,'MSEs');
            nitr = get(stepMonitor,'nItr');
            mseActual = mses(nitr);
            
            % Evaluation
            testCase.assertEqual(mseActual,mseExpctd)
            
        end
        
        % Test denoising
        function testIstaUdHaarDeNoise(testCase)
            
            % Preperation
            srcImg = rand(32,32,64);
            import saivdr.dictionary.udhaar.*
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer = UdHaarAnalysis3dSystem();
            analyzer.NumberOfLevels = 3;            
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
            import saivdr.restoration.ista.*
            testCase.istaimrstr = IstaImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'Lambda',1e-2);
            
            % MSE after processing
            resImg = step(testCase.istaimrstr,obsImg);
            
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
                import saivdr.restoration.ista.*
                testCase.istaimrstr = IstaImRestoration3d(...
                    'AdjOfSynthesizer',analyzer,...
                    'LinearProcess',linearProcess);
            catch
            end
            
            % Abset ajoint of synthesizer
            try
                testCase.istaimrstr = IstaImRestoration3d(...
                    'Synthesizer',synthesizer,...
                    'LinearProcess',linearProcess);
            catch
            end
            
            % Abset ajoint of synthesizer
            try
                testCase.istaimrstr = IstaImRestoration3d(...
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
            import saivdr.restoration.ista.*
            
            
            % Abset synthesizer
            messageExpctd = 'Synthesizer must be given.';
            try
                testCase.istaimrstr = IstaImRestoration3d(...
                    'AdjOfSynthesizer',analyzer,...
                    'LinearProcess',   linearProcess);
                step(testCase.istaimrstr,[]);
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
                testCase.istaimrstr = IstaImRestoration3d(...
                    'Synthesizer',   synthesizer,...
                    'LinearProcess', linearProcess);
                step(testCase.istaimrstr,[]);
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
                testCase.istaimrstr = IstaImRestoration3d(...
                    'AdjOfSynthesizer',analyzer,...
                    'Synthesizer',   synthesizer);
                step(testCase.istaimrstr,[]);
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
            analyzer.NumberOfLevels = 3;            
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
            import saivdr.restoration.ista.*
            testCase.istaimrstr = IstaImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'Lambda',1e-2);
            
            % MSE by original object
            resImg = step(testCase.istaimrstr,obsImg);
            mseOrg = mse(resImg,srcImg);
            
            % Instantiation of target class
            cloneIstaimrstr = clone(testCase.istaimrstr);
            
            % MSE by clone object
            resImg = step(cloneIstaimrstr,obsImg);
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
            analyzer.NumberOfLevels = 3;            
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
            import saivdr.restoration.ista.*
            testCase.istaimrstr = IstaImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'Lambda',1e-2);
            
            % MSE by original object
            resImg = step(testCase.istaimrstr,obsImg);
            mseOrg = mse(resImg,srcImg);
            
            % Instantiation of target class
            cloneIstaimrstr = clone(testCase.istaimrstr);
            
            % MSE by clone object
            resImg = step(cloneIstaimrstr,obsImg);
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
            analyzer.NumberOfLevels = 3;            
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = BlurSystem(...
                'DataType','Volumetric Data');
            
            % Instantiation of target class
            import saivdr.restoration.ista.*
            testCase.istaimrstr = IstaImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'Lambda',1e-2);
            
            % Instantiation of target class
            cloneIstaimrstr = clone(testCase.istaimrstr);
            
            % Evaluation
            testCase.verifyEqual(cloneIstaimrstr,testCase.istaimrstr);
            testCase.verifyFalse(cloneIstaimrstr == testCase.istaimrstr);
            %
            prpOrg = get(testCase.istaimrstr,'Synthesizer');
            prpCln = get(cloneIstaimrstr,'Synthesizer');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            prpOrg = get(testCase.istaimrstr,'AdjOfSynthesizer');
            prpCln = get(cloneIstaimrstr,'AdjOfSynthesizer');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            prpOrg = get(testCase.istaimrstr,'LinearProcess');
            prpCln = get(cloneIstaimrstr,'LinearProcess');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            
        end
        
        % Test
        function testCloneWithUdHaarChildObj(testCase)
            
            % Preperation
            import saivdr.dictionary.udhaar.*
            synthesizer = UdHaarSynthesis3dSystem();
            analyzer = UdHaarAnalysis3dSystem();
            analyzer.NumberOfLevels = 3;
            import saivdr.degradation.*
            import saivdr.degradation.linearprocess.*
            import saivdr.degradation.noiseprocess.*
            linearProcess = BlurSystem(...
                'DataType','Volumetric Data');
            
            % Instantiation of target class
            import saivdr.restoration.ista.*
            testCase.istaimrstr = IstaImRestoration3d(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'LinearProcess',linearProcess,...
                'Lambda',1e-2);
            
            % Instantiation of target class
            cloneIstaimrstr = clone(testCase.istaimrstr);
            
            % Evaluation
            testCase.verifyEqual(cloneIstaimrstr,testCase.istaimrstr);
            testCase.verifyFalse(cloneIstaimrstr == testCase.istaimrstr);
            %
            prpOrg = get(testCase.istaimrstr,'Synthesizer');
            prpCln = get(cloneIstaimrstr,'Synthesizer');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            prpOrg = get(testCase.istaimrstr,'AdjOfSynthesizer');
            prpCln = get(cloneIstaimrstr,'AdjOfSynthesizer');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            prpOrg = get(testCase.istaimrstr,'LinearProcess');
            prpCln = get(cloneIstaimrstr,'LinearProcess');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            
        end

    end
    
end

