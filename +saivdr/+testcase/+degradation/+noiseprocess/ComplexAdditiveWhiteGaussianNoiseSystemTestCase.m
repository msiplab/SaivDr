classdef ComplexAdditiveWhiteGaussianNoiseSystemTestCase < matlab.unittest.TestCase
    %ADDITIVEWHITEGAUSSIANNOISSYSTEMTESTCASE Test case for ComplexAdditiveWhiteGaussianNoiseSystem
    %
    % SVN identifier:
    % $Id: ComplexAdditiveWhiteGaussianNoiseSystemTestCase.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627    
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
            testCase.noiseproc = ComplexAdditiveWhiteGaussianNoiseSystem();

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
            testCase.noiseproc = ComplexAdditiveWhiteGaussianNoiseSystem(...
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
    end
    
end

