classdef GaussianDenoiserSfthTestCase < matlab.unittest.TestCase
    %GAUSSIANDENOISERSFTHTESTCASE Gaussian denoiser with soft thresholding
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
        type = {'single','double'};
        inputSize = struct('small',8, 'medium', 16, 'large', 32);
        sigma     = struct('small',0.1,'large',2);
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
        
        function testConstruction(testCase,sigma)
            
            sigmaExpctd = sigma;
            
            import saivdr.restoration.denoiser.*
            testCase.target = GaussianDenoiserSfth('Sigma',sigmaExpctd);
            
            sigmaActual = testCase.target.Sigma;
            
            testCase.verifyEqual(sigmaActual,sigmaExpctd);
            
        end

        function testStepScalar(testCase,inputSize,sigma)
            
            x = randn(inputSize,1);
            
            v = max(abs(x)-sigma^2,0);
            yExpctd = sign(x).*v;
            
            import saivdr.restoration.denoiser.*            
            testCase.target = GaussianDenoiserSfth('Sigma',sigma);
            
            yActual = testCase.target.step(x);
            
            testCase.verifyEqual(yActual,yExpctd);
            
        end
        
        function testSetSigma(testCase,inputSize,sigma)
            
            x = randn(inputSize,1);
            
            v = max(abs(x)-sigma^2,0);
            yExpctd = sign(x).*v;
            
            import saivdr.restoration.denoiser.*        
            testCase.target = GaussianDenoiserSfth();
            testCase.target.Sigma = sigma;
            
            yActual = testCase.target.step(x);
            
            testCase.verifyEqual(yActual,yExpctd);
            
        end
        
        function testStepVector(testCase,inputSize,sigma)
            
            x    = sigma*randn(inputSize,1);
            svec = sigma*rand(inputSize,1);
            
            v = abs(x)-svec.^2;
            v(v<0) = 0;
            yExpctd = sign(x).*v;
        
            import saivdr.restoration.denoiser.*            
            testCase.target = GaussianDenoiserSfth('Sigma',svec);
            
            yActual = testCase.target.step(x);
            
            testCase.verifyEqual(yActual,yExpctd);
            
        end
       
    end

end

