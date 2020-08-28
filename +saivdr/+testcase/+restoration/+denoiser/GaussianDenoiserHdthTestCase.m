classdef GaussianDenoiserHdthTestCase < matlab.unittest.TestCase
    %TESTCASEPLGHARDTHRESHOLDING Gaussian denoiser with hard thresholding
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
        mode = {'Regularization','Hard Constraint'};
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
            testCase.target = GaussianDenoiserHdth('Sigma',sigmaExpctd);
            
            sigmaActual = testCase.target.Sigma;
            
            testCase.verifyEqual(sigmaActual,sigmaExpctd);
            
        end

        function testStepScalar(testCase,inputSize,sigma)
            
            x = randn(inputSize,1);
            
            yExpctd = x;
            yExpctd(abs(x)<=sigma^2) = 0;
            
            import saivdr.restoration.denoiser.*            
            testCase.target = GaussianDenoiserHdth('Sigma',sigma);
            
            yActual = testCase.target.step(x);
            
            testCase.verifyEqual(yActual,yExpctd,'AbsTol',1e-10);
            
        end
        
        function testSetSigma(testCase,inputSize,sigma)
            
            x = randn(inputSize,1);
            
            yExpctd = x;
            yExpctd(abs(x)<=sigma^2) = 0;
            
            import saivdr.restoration.denoiser.*        
            testCase.target = GaussianDenoiserHdth();
            testCase.target.Sigma = sigma;
            
            yActual = testCase.target.step(x);
            testCase.verifyEqual(yActual,yExpctd,'AbsTol',1e-10);
            
            yExpctd(abs(x)<=(2*sigma)^2) = 0;
            testCase.target.Sigma = 2*sigma;
            yActual = testCase.target.step(x);
            testCase.verifyEqual(yActual,yExpctd,'AbsTol',1e-10);
            
        end
        
        function testStepVector(testCase,inputSize,sigma)
            
            x    = sigma*randn(inputSize,1);
            svec = sigma*rand(inputSize,1);
            
            yExpctd = x;
            yExpctd((abs(x)-svec.^2)<=0) = 0;
        
            import saivdr.restoration.denoiser.*            
            testCase.target = GaussianDenoiserHdth('Sigma',svec);
            
            yActual = testCase.target.step(x);
            
            testCase.verifyEqual(yActual,yExpctd,'AbsTol',1e-10);
            
        end
    end
end
        