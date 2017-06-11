classdef PoissonNoiseSystemTestCase < matlab.unittest.TestCase
    %POISSONNOISSYSTEMTESTCASE Test case for PoissonNoiseSystem
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2017, Shogo MURAMATSU
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
        noiseproc
    end
    
    methods (TestMethodTeardown)
        function deleteObject(testCase)
            delete(testCase.noiseproc);
        end
    end
    
    methods (Test)
        
        function testPoissonNoiseLambda0_5(testCase)

            % Preparation
            lambda = 0.5;
            srcImg = lambda*ones(128);  
            
            % Expected values
            meanExpctd = lambda;
            varExpctd  = lambda*1e-12;
            
            % Instantiation of target class
            import saivdr.degradation.noiseprocess.*
            testCase.noiseproc = PoissonNoiseSystem();

            % Actual values
            resImg = step(testCase.noiseproc,srcImg);
            meanActual = mean(resImg(:));
            varActual  = var(resImg(:));
            
            % Evaluation
            diff = meanExpctd - meanActual;
            testCase.verifyEqual(meanActual,meanExpctd,'RelTol',5e-2,sprintf('%g',diff));
            diff = varExpctd - varActual;
            testCase.verifyEqual(varActual,varExpctd,'RelTol',5e-2,sprintf('%g',diff));                        
            
        end
        
        function testPoissonNoiseLambda0_25(testCase)

            % Preparation
            lambda = 0.25;
            srcImg = lambda*ones(128);  
            
            % Expected values
            meanExpctd = lambda;
            varExpctd  = lambda*1e-12;
            
            % Instantiation of target class
            import saivdr.degradation.noiseprocess.*
            testCase.noiseproc = PoissonNoiseSystem();

            % Actual values
            resImg = step(testCase.noiseproc,srcImg);
            meanActual = mean(resImg(:));
            varActual  = var(resImg(:));
            
            % Evaluation
            diff = meanExpctd - meanActual;
            testCase.verifyEqual(meanActual,meanExpctd,'RelTol',5e-2,sprintf('%g',diff));
            diff = varExpctd - varActual;
            testCase.verifyEqual(varActual,varExpctd,'RelTol',5e-2,sprintf('%g',diff));                        
            
        end        

        function testPoissonNoiseLambda0_75(testCase)

            % Preparation
            lambda = 0.25;
            srcImg = lambda*ones(128);  
            
            % Expected values
            meanExpctd = lambda;
            varExpctd  = lambda*1e-12;
            
            % Instantiation of target class
            import saivdr.degradation.noiseprocess.*
            testCase.noiseproc = PoissonNoiseSystem();
            
            % Actual values
            resImg = step(testCase.noiseproc,srcImg);
            meanActual = mean(resImg(:));
            varActual  = var(resImg(:));
            
            % Evaluation
            diff = meanExpctd - meanActual;
            testCase.verifyEqual(meanActual,meanExpctd,'RelTol',5e-2,sprintf('%g',diff));
            diff = varExpctd - varActual;
            testCase.verifyEqual(varActual,varExpctd,'RelTol',5e-2,sprintf('%g',diff));
            
        end
        
        function testPoissonNoiseLambda0_5VolumetricData(testCase)
            
            % Preparation
            lambda = 0.5;
            srcImg = lambda*ones(64,64,64);
            
            % Expected values
            meanExpctd = lambda;
            varExpctd  = lambda*1e-12;
            
            % Instantiation of target class
            import saivdr.degradation.noiseprocess.*
            testCase.noiseproc = PoissonNoiseSystem();
            
            % Actual values
            resImg = step(testCase.noiseproc,srcImg);
            meanActual = mean(resImg(:));
            varActual  = var(resImg(:));
            
            % Evaluation
            diff = meanExpctd - meanActual;
            testCase.verifyEqual(meanActual,meanExpctd,'RelTol',5e-2,sprintf('%g',diff));
            diff = varExpctd - varActual;
            testCase.verifyEqual(varActual,varExpctd,'RelTol',5e-2,sprintf('%g',diff));
            
        end
        
    end
    
end

