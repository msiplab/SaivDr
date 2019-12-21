classdef GaussianDenoiserBm4dTestCase < matlab.unittest.TestCase
    %GAUSSIANDENOISERBM4DTESTCASE Gaussian denoiser with BM4D
    %
    % Requirements: MATLAB R2019b
    %
    % Copyright (c) 2019-, Shogo MURAMATSU
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
        %type = {'single','double','uint16'};
        dim1 = struct('small',8, 'medium', 16, 'large', 32);
        dim2 = struct('small',8, 'medium', 16, 'large', 32);
        dim3 = struct('small',8, 'medium', 16, 'large', 32);
        sigma = struct('small',0.1,'large',2);
    end
    
    methods (Test)
        
        function testConstruction(testCase,sigma)
            
            sigmaExpctd = sigma;
            
            import saivdr.restoration.denoiser.*            
            target = GaussianDenoiserBm4d('Sigma',sigmaExpctd);
            
            sigmaActual = target.Sigma;
            
            testCase.verifyEqual(sigmaActual,sigmaExpctd);
            
        end
        
        function testStep(testCase,dim1,dim2,dim3,sigma)
            
            x = randn(dim1,dim2,dim3);
                      
            import saivdr.restoration.denoiser.*                        
            target = GaussianDenoiserBm4d('Sigma',sigma);
            
            yActual = target.step(x);
            
            testCase.verifySize(yActual,size(x))
            
        end
        
         function testSigma(testCase,dim1,dim2,dim3,sigma)

             sigmaExpctd = sigma;
             
             x = randn(dim1,dim2,dim3);
                      
             import saivdr.restoration.denoiser.*                                     
             target = GaussianDenoiserBm4d();
             target.Sigma = sigma;
            
             target.step(x);
             
             sigmaActual = target.Sigma;
            
             testCase.verifyEqual(sigmaActual,sigmaExpctd);
            
        end
        
    end
end

