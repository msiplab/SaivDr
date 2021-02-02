classdef AdditiveWhiteGaussianNoiseSystemTestCase < matlab.unittest.TestCase
    %ADDITIVEWHITEGAUSSIANNOISSYSTEMTESTCASE Test case for AdditiveWhiteGaussianNoiseSystem
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
        
        function testAwgnDefault(testCase)
            
            % Preparation
            srcImg = 0.5*ones(128);
            
            % Expected values
            meanExpctd = 0;
            varExpctd  = 0.01;
            
            % Instantiation of target class
            import saivdr.degradation.noiseprocess.*
            testCase.noiseproc = AdditiveWhiteGaussianNoiseSystem();

            % Actual values
            resImg = step(testCase.noiseproc,srcImg);
            noiseImg = resImg - srcImg;
            meanActual = mean(noiseImg(:));
            varActual  = var(noiseImg(:));
            
            % Evaluation
            diff = meanExpctd - meanActual;
            testCase.verifyEqual(meanActual,meanExpctd,'AbsTol',1e-2,sprintf('%g',diff));
            diff = varExpctd - varActual;
            testCase.verifyEqual(varActual,varExpctd,'AbsTol',1e-2,sprintf('%g',diff));            
            
        end
        
        function testAwgnMean0Var0_02(testCase)

            % Preparation
            srcImg = 0.5*ones(128);
            
            % Expected values
            meanExpctd = 0;
            varExpctd  = 0.02;
            
            % Instantiation of target class
            import saivdr.degradation.noiseprocess.*
            testCase.noiseproc = AdditiveWhiteGaussianNoiseSystem(...
                'Mean',meanExpctd,...
                'Variance',varExpctd);

            % Actual values
            resImg = step(testCase.noiseproc,srcImg);
            noiseImg = resImg - srcImg;
            meanActual = mean(noiseImg(:));
            varActual  = var(noiseImg(:));
            
            % Evaluation
            diff = meanExpctd - meanActual;
            testCase.verifyEqual(meanActual,meanExpctd,'AbsTol',1e-2,sprintf('%g',diff));
            diff = varExpctd - varActual;
            testCase.verifyEqual(varActual,varExpctd,'AbsTol',1e-2,sprintf('%g',diff));            
            
        end        
        
        function testAwgnVolumetricData(testCase)
            
            % Preparation
            srcImg = 0.5*ones(64,64,64);
            
            % Expected values
            meanExpctd = 0;
            varExpctd  = 0.01;
            
            % Instantiation of target class
            import saivdr.degradation.noiseprocess.*
            testCase.noiseproc = AdditiveWhiteGaussianNoiseSystem();
            
            % Actual values
            resImg = step(testCase.noiseproc,srcImg);
            noiseImg = resImg - srcImg;
            meanActual = mean(noiseImg(:));
            varActual  = var(noiseImg(:));
            
            % Evaluation
            diff = meanExpctd - meanActual;
            testCase.verifyEqual(meanActual,meanExpctd,'AbsTol',1e-2,sprintf('%g',diff));
            diff = varExpctd - varActual;
            testCase.verifyEqual(varActual,varExpctd,'AbsTol',1e-2,sprintf('%g',diff));
            
        end
        
    end
    
end

