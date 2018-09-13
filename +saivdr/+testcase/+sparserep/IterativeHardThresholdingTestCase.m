classdef IterativeHardThresholdingTestCase < ...
        matlab.unittest.TestCase
    %ITERATIVEHARDTHRESHOLDINGTESTCASE Test case for IterativeHardThresholding
    %
    % SVN identifier:
    % $Id: IterativeHardThresholdingTestCase.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
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
    % http://msiplab.eng.niigata-u.ac.jp/    
    %
    properties
        iht
    end
        
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.iht);
        end
    end
    
    methods (Test)
        
        function testIhtNsoltCh5plus2Ord44Level1(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            height = 32;
            width  = 32;
            srcImg = rand(height,width);
            
            % Expected values
            nCoefsExpctd = 32;
            sizeOfCoefExpctd = [ 1 numel(srcImg)*sum(nChs)/prod(nDecs) ];
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Instantiation of target class
            import saivdr.sparserep.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');            
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual, scalesActual] = ...
                testCase.iht.step(srcImg);
            nCoefsActual = nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, sizeOfCoefExpctd);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,scalesActual);
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
            %}
        end

        function testIhtNsoltCh5plus2Ord44Level2(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            nLevels = 2;
            height = 32;
            width  = 32;
            srcImg = rand(height,width);
            
            % Expected values
            nCoefsExpctd = 32;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Instantiation of target class
            import saivdr.sparserep.*
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');                        
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);                                                            
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Expected values
            nCoefs = numel(srcImg)*((sum(nChs)-1)/prod(nDecs) ...
                + sum(nChs)/prod(nDecs)^2);
            sizeOfCoefsExpctd = [ 1 nCoefs ];
            
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual, scalesActual] = ...
                testCase.iht.step(srcImg);
            nCoefsActual =  nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, sizeOfCoefsExpctd);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,scalesActual);
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
            %}        
        end

        function testIhtNsoltCh5plus2Ord44Level3(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            nLevels = 3;
            height = 32;
            width  = 32;
            srcImg = rand(height,width);
            
            % Expected values
            nCoefsExpctd = 32;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds, ...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Instantiation of target class
            import saivdr.sparserep.*            
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');                        
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);                                                            
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,....
                'AdjOfSynthesizer',analyzer);
            
            % Expected values
            nCoefs = numel(srcImg)*((sum(nChs)-1)/prod(nDecs) ...
                + (sum(nChs)-1)/prod(nDecs)^2 ...
                + sum(nChs)/prod(nDecs)^3);
            sizeOfCoefsExpctd = [ 1 nCoefs ];
            
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual, scalesActual] = ...
                testCase.iht.step(srcImg);
            nCoefsActual = nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, sizeOfCoefsExpctd);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,scalesActual);
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
                %}
        end

        function testIhtNsoltCh2plus2Ord22Level1(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 4 4 ];
            height = 32;
            width  = 32;
            srcImg = rand(height,width);
            
            % Expected values
            nCoefsExpctd = 32;
            sizeOfCoefsExpctd = [ 1 numel(srcImg)*sum(nChs)/prod(nDecs) ];
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Instantiation of target class
            import saivdr.sparserep.*            
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination');
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual, scalesActual] = ...
                testCase.iht.step(srcImg);
            nCoefsActual = nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, sizeOfCoefsExpctd);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,scalesActual);
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
            %}
        end

        function testIhtNsoltCh2plus2Ord22Level2(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            nLevels = 2;
            height = 32;
            width  = 32;
            srcImg = rand(height,width);
            
            % Expected values
            nCoefsExpctd = 32;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Instantiation of target class
            import saivdr.sparserep.*            
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');            
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);                                    
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Expected values
            nCoefs = numel(srcImg)*(...
                (sum(nChs)-1)/prod(nDecs) + sum(nChs)/(prod(nDecs)^2));
            sizeOfCoefsExpctd = [ 1 nCoefs ];
            
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual, scalesActual] = ...
                testCase.iht.step(srcImg);
            nCoefsActual = nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, sizeOfCoefsExpctd);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,scalesActual);
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
            %}        
        end
       
        function testIhtNsoltCh2plus2Ord22Level3(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];
            nLevels = 3;
            height = 32;
            width  = 32;
            srcImg = rand(height,width);
            
            % Expected values
            nCoefsExpctd = 32;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...              
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Instantiation of target class
            import saivdr.sparserep.*            
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');            
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);                                                
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Expected values
            nCoefs = numel(srcImg)*sum(nChs)/prod(nDecs);
            sizeOfCoefsExpctd = [ 1 nCoefs ];
                
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual, scalesActual] = ...
                testCase.iht.step(srcImg);
            nCoefsActual = nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, sizeOfCoefsExpctd);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,scalesActual);
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
                %}
        end        
   
        function testIhtNsoltCh5plus2Ord44Level3Aprx(testCase)
            
            nDecs = [ 2 2 ];
            nChs  = [ 5 2 ];
            nOrds = [ 4 4 ];
            height = 32;
            width  = 64;
            nLevels = 3;
            srcImg = rand(height,width);
            
            % Preparation
            vm = 1;
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder', nOrds,...
                'NumberOfVanishingMoments', vm,...
                'OutputMode', 'ParameterMatrixSet');
            synthesizer = NsoltFactory.createSynthesis2dSystem(...
                lppufb,'BoundaryOperation','Termination');            
            analyzer    = NsoltFactory.createAnalysis2dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);                                                

            % Instantiation of target class
            import saivdr.sparserep.*           
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.iht.NumberOfCoefficients = 1;
            resid0 = testCase.iht.step(srcImg);
            for nCoefs = 2:32
                testCase.iht.NumberOfCoefficients = nCoefs;
                resid1 = testCase.iht.step(srcImg);
                testCase.verifyThat(norm(resid1(:)) < norm(resid0(:)), ...
                    IsTrue, ...
                    sprintf('||r0||^2 = %g must be less than ||r1||^2 = %g.',...
                    norm(resid0(:)),norm(resid1(:))));
                resid0 = resid1;
            end
        end        

        function testIhtNsHaarLevel1(testCase)
            
            nLevels = 1;
            height = 32;
            width  = 32;
            srcImg = rand(height,width);
            
            % Expected values
            nCoefsExpctd = 32;
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            synthesizer = UdHaarSynthesis2dSystem();
            analyzer    = UdHaarAnalysis2dSystem('NumberOfLevels',nLevels);
            import saivdr.sparserep.*
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
                
            
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual] = testCase.iht.step(srcImg);
            nCoefsActual = nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, [ 1 (3*nLevels+1)*height*width]);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,repmat(size(srcImg),[4 1]));
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
            %}
        end
                
        function testIhtNsHaarLevel2(testCase)
            
            nLevels = 2;
            height = 32;
            width  = 32;
            srcImg = rand(height,width);
            
            % Expected values
            nCoefsExpctd = 32;
            
            % Instantiation of target class
            import saivdr.dictionary.udhaar.*
            import saivdr.sparserep.*
            synthesizer = UdHaarSynthesis2dSystem();
            analyzer    = UdHaarAnalysis2dSystem(...
                'NumberOfLevels',nLevels);
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual] = testCase.iht.step(srcImg);
            nCoefsActual = nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, [ 1 (3*nLevels+1)*height*width]);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,repmat(size(srcImg),[7 1]));
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
            %}
        end
        
        % Test step monitoring
        function testStepMonitoring(testCase)
            
            % Preperation
            height = 32;
            width  = 32;
            srcImg = rand(height,width);
            nCoefs = 32;
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis2dSystem();
            analyzer    = NsoltFactory.createAnalysis2dSystem();
            analyzer.NumberOfLevels = 3;
            
            % Instantiation of step monitor
            import saivdr.utility.*
            stepMonitor = StepMonitoringSystem(...
                'SourceImage',srcImg,...
                'IsVerbose',false,...
                'IsMSE',true);
            
            % Instantiation of target class
            import saivdr.sparserep.*
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer,...
                'StepMonitor',stepMonitor);
            
            % Definition of MSE
            mse = @(x,y) sum((double(x(:))-double(y(:))).^2)/numel(x);
            
            % MSE after processing
            testCase.iht.NumberOfCoefficients = nCoefs;
            resImg = testCase.iht.step(srcImg);
            recImg = srcImg - resImg;
            mseExpctd = mse(uint8(255*recImg),uint8(255*srcImg));
            
            % Actual value
            mses = get(stepMonitor,'MSEs');
            nitr = get(stepMonitor,'nItr');
            mseActual = mses(nitr);
            
            % Evaluation
            diff = max(abs(mseExpctd(:)-mseActual(:))./abs(mseExpctd(:)));
            testCase.assertEqual(mseActual,mseExpctd,'RelTol',1e-10,...
                sprintf('diff = %f\n',diff))
            
        end
                
        % Test
        function testClone(testCase)
            
            % Preperation
            height = 32;
            width  = 32;
            nCoefs = 32;
            srcImg = rand(height,width);
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis2dSystem();
            analyzer    = NsoltFactory.createAnalysis2dSystem();
            analyze.NumberOfLevels = 3;
            
            % MSE before processing
            mse = @(x,y) sum((x(:)-y(:)).^2)/numel(x);
            
            % Instantiation of target class
            import saivdr.sparserep.*
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % MSE by original object
            testCase.iht.NumberOfCoefficients = nCoefs;
            resImg = testCase.iht.step(srcImg);
            recImg = srcImg - resImg;
            mseOrg = mse(int16(255*recImg),int16(255*srcImg));            
            
            % Instantiation of target class
            cloneIht = clone(testCase.iht);
            
            % MSE by clone object
            cloneIht.NumberOfCoefficients = nCoefs;
            resImg = cloneIht.step(srcImg);
            recImg = srcImg - resImg;
            mseCln = mse(int16(255*recImg),int16(255*srcImg));                        
            
            % Evaluation
            testCase.verifyEqual(mseCln,mseOrg);
            
        end
        
        % Test
        function testCloneWithChildObj(testCase)
            
            % Preperation
            import saivdr.dictionary.nsoltx.*
            synthesizer = NsoltFactory.createSynthesis2dSystem();
            analyzer    = NsoltFactory.createAnalysis2dSystem();
            analyzer.NumberOfLevels = 3;
            
            % Instantiation of target class
            import saivdr.sparserep.*
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Instantiation of target class
            cloneIht = clone(testCase.iht);
            
            % Evaluation
            testCase.verifyEqual(cloneIht,testCase.iht);
            testCase.verifyFalse(cloneIht == testCase.iht);
            %
            prpOrg = get(testCase.iht,'Synthesizer');
            prpCln = get(cloneIht,'Synthesizer');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            prpOrg = get(testCase.iht,'AdjOfSynthesizer');
            prpCln = get(cloneIht,'AdjOfSynthesizer');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            
        end        

        function testIhtNsoltDec222Ch66Ord222Level1(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 6 ];
            nOrds = [ 2 2 2 ];
            height = 32;
            width  = 32;
            depth  = 32;
            srcImg = rand(height,width,depth);
            
            % Expected values
            nCoefsExpctd = 32;
            sizeOfCoefsExpctd = [ 1 numel(srcImg)*sum(nChs)/prod(nDecs) ];
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Instantiation of target class
            import saivdr.sparserep.*            
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination');            
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual, scalesActual] = ...
                testCase.iht.step(srcImg);
            nCoefsActual = nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, sizeOfCoefsExpctd);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,scalesActual);
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
            %}
        end

        function testIhtNsoltDec222Ch66Ord222Level2(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 6 ];
            nOrds = [ 2 2 2 ];
            nLevels = 2;
            height = 32;
            width  = 32;
            depth  = 32;
            srcImg = rand(height,width,depth);
            
            % Expected values
            nCoefsExpctd = 32;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Instantiation of target class
            import saivdr.sparserep.*            
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');            
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Expected values
            P = sum(nChs);
            M = prod(nDecs);
            nCoefs = numel(srcImg)*((P-1)/(M-1)-(P-M)/((M-1)*M^nLevels));
            sizeOfCoefsExpctd = [ 1 nCoefs ];
            
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual, scalesActual] = ...
                testCase.iht.step(srcImg);
            nCoefsActual = nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, sizeOfCoefsExpctd);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,scalesActual);
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
            %}        
        end
       
        function testIhtNsoltDec222Ch66Ord222Level3(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 6 ];
            nOrds = [ 2 2 2 ];
            nLevels = 3;
            height = 32;
            width  = 32;
            depth  = 32;
            srcImg = rand(height,width,depth);
            
            % Expected values
            nCoefsExpctd = 32;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...              
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Instantiation of target class
            import saivdr.sparserep.*            
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');            
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Expected values
            P = sum(nChs);
            M = prod(nDecs);
            nCoefs = numel(srcImg)*(...
                (P-1)/(M-1)-(P-M)/((M-1)*M^nLevels));
            sizeOfCoefsExpctd = [ 1 nCoefs ];
            
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual, scalesActual] = ...
                testCase.iht.step(srcImg);
            nCoefsActual = nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, sizeOfCoefsExpctd);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,scalesActual);
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
                %}
        end        
        
        function testIhtNsoltDec222Ch64Ord222Level3(testCase)
            
            nDecs = [ 2 2 2 ];
            nChs  = [ 6 4 ];
            nOrds = [ 2 2 2 ];
            nLevels = 3;
            height = 16;
            width  = 32;
            depth  = 64;
            srcImg = rand(height,width,depth);
            
            % Expected values
            nCoefsExpctd = 32;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...              
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Instantiation of target class
            import saivdr.sparserep.*            
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');            
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Expected values
            P = sum(nChs);
            M = prod(nDecs);
            nCoefs = numel(srcImg)*(...
                (P-1)/(M-1)-(P-M)/((M-1)*M^nLevels));
            sizeOfCoefsExpctd = [ 1 nCoefs ];
            
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual, scalesActual] = ...
                testCase.iht.step(srcImg);
            nCoefsActual = nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, sizeOfCoefsExpctd);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,scalesActual);
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
                %}
        end                

        function testIhtNsoltDec112Ch22Ord222Level3(testCase)
            
            nDecs = [ 1 1 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 2 ];
            nLevels = 3;
            height = 16;
            width  = 32;
            depth  = 64;
            srcImg = rand(height,width,depth);
            
            % Expected values
            nCoefsExpctd = 32;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb3dSystem(...              
                'DecimationFactor', nDecs,...
                'NumberOfChannels', nChs ,...
                'PolyPhaseOrder', nOrds,...
                'OutputMode','ParameterMatrixSet');
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Instantiation of target class
            import saivdr.sparserep.*            
            synthesizer = NsoltFactory.createSynthesis3dSystem(...
                lppufb,'BoundaryOperation','Termination');            
            analyzer    = NsoltFactory.createAnalysis3dSystem(...
                lppufb,'BoundaryOperation','Termination',...
                'NumberOfLevels',nLevels);
            testCase.iht = IterativeHardThresholding(...
                'Synthesizer',synthesizer,...
                'AdjOfSynthesizer',analyzer);
            
            % Expected values
            P = sum(nChs);
            M = prod(nDecs);
            nCoefs = numel(srcImg)*(...
                (P-1)/(M-1)-(P-M)/((M-1)*M^nLevels));
            sizeOfCoefsExpctd = [ 1 nCoefs ];
            
            % Actual values
            testCase.iht.NumberOfCoefficients = nCoefsExpctd;
            [residActual, coefsActual, scalesActual] = ...
                testCase.iht.step(srcImg);
            nCoefsActual = nnz(coefsActual);
            
            % Evaluation
            import matlab.unittest.constraints.IsTrue;
            testCase.verifySize(residActual, size(srcImg));
            testCase.verifySize(coefsActual, sizeOfCoefsExpctd);
            testCase.verifyThat(norm(residActual(:)) < norm(srcImg(:)), ...
                IsTrue, ...
                sprintf('||r||^2 = %g must be less than ||x||^2 = %g.',...
                norm(residActual(:)),norm(srcImg(:)))); 
            testCase.verifyThat(nCoefsActual<=nCoefsExpctd,IsTrue);
            
            resImg = step(synthesizer,coefsActual,scalesActual);
            testCase.verifyEqual(resImg(:)+residActual(:), srcImg(:),...
                'RelTol',1e-10);
            
            %{
            mse = @(x,y) ((x(:)-y(:)).^2)/numel(x);
            psnr = @(x,y) -10*log10(mse(x,y));
            sprintf('psnr = %6.2f [dB]\n',psnr(resImg,srcImg))
                %}
        end    
    end
    
end
