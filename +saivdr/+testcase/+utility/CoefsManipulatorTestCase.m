classdef CoefsManipulatorTestCase < matlab.unittest.TestCase
    %COEFSMANIPULATORTESTCASE Test cases for CoefsManipulator  
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
        width = struct('small', 64, 'medium', 96, 'large', 128);
        height = struct('small', 64, 'medium', 96, 'large', 128);
        depth = struct('small', 64, 'medium', 96, 'large', 128);        
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
        
        function testDefaultStep(testCase,width,height)
            
            % Parameters
            coefspre = randn(width,height);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator();
            
            % Expected value
            coefsExpctd = coefspre;
            
            % Actual value
            coefsActual = testCase.target.step(coefspre);
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
        
        function testSoftThresholding2d(testCase,width,height)
            
            % Parameters
            coefspre = randn(width,height);
            
            % Function
            lambda = 1e-3;
            g = @(x) sign(x).*((abs(x)-lambda)+abs(abs(x)-lambda))/2;            
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator();
            
            % Expected value
            coefsExpctd = g(coefspre);
            
            % Actual value
            coefsActual = testCase.target.step(g,coefspre);
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end        

        
        function testSoftThresholding3d(testCase,width,height,depth)
            
            % Parameters
            coefspre = randn(width,height,depth);
            
            % Function
            lambda = 1e-3;
            g = @(x) sign(x).*((abs(x)-lambda)+abs(abs(x)-lambda))/2;            
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator();
            
            % Expected value
            coefsExpctd = g(coefspre);
            
            % Actual value
            coefsActual = testCase.target.step(g,coefspre);
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end        

        function testIterativeSoftThresholding2d(testCase,width,height)
            
            % Parameters
            nIters = 5;
            coefs  = cell(nIters+1,1);
            for iIter = 1:nIters+1
                coefs{iIter} = randn(width,height);
            end
            
            % Function
            lambda = 1e-3;
            gamma  = 1e-3;
            f = @(x,xpre) xpre-gamma*x;
            g = @(x) sign(x).*((abs(x)-lambda)+abs(abs(x)-lambda))/2;
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator();
            
            % Expected value
            xpre = coefs{1};
            for iIter = 1:nIters
                v = f(coefs{iIter+1},xpre);
                xpre = g(v);
            end
            interExpctd = v;
            coefsExpctd = xpre;
            
            % Actual value
            xpre = coefs{1};
            for iIter = 1:nIters
                [xpre,v] = testCase.target.step(g,@(x) f(x,xpre),...
                    coefs{iIter+1});
            end
            interActual = v;
            coefsActual = xpre;
            
            % Evaluation
            testCase.verifySize(interActual,size(interExpctd));            
            diff = max(abs(interExpctd(:) - interActual(:)));
            testCase.verifyEqual(interActual,interExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end      
                
        function testIterativeSoftThresholding3d(testCase,width,height,depth)
            
            % Parameters
            nIters = 5;
            coefs  = cell(nIters+1,1);
            for iIter = 1:nIters+1
                coefs{iIter} = randn(width,height,depth);
            end
            
            % Function
            lambda = 1e-3;
            gamma  = 1e-3;
            f = @(x,xpre) xpre-gamma*x;
            g = @(x) sign(x).*((abs(x)-lambda)+abs(abs(x)-lambda))/2;
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator();
            
            % Expected value
            xpre = coefs{1};
            for iIter = 1:nIters
                v = f(coefs{iIter+1},xpre);
                xpre = g(v);
            end
            interExpctd = v;
            coefsExpctd = xpre;
            
            % Actual value
            xpre = coefs{1};
            for iIter = 1:nIters
                [xpre,v] = testCase.target.step(g,@(x) f(x,xpre),...
                    coefs{iIter+1});
            end
            interActual = v;
            coefsActual = xpre;
            
            % Evaluation
            testCase.verifySize(interActual,size(interExpctd));            
            diff = max(abs(interExpctd(:) - interActual(:)));
            testCase.verifyEqual(interActual,interExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end      
        
        function testPdsHsHcOct3d(testCase,width,height,depth)
            
            % Parameters
            nIters = 5;
            coefs  = cell(nIters+1,1);
            for iIter = 1:nIters+1
                coefs{iIter} = randn(width,height,depth);
            end
            
            % Function
            lambda = 1e-3;
            gamma  = 1e-3;
            f = @(x,xpre) xpre-gamma*x;
            g = @(x) sign(x).*((abs(x)-lambda)+abs(abs(x)-lambda))/2;
            h = @(x,xpre) 2*x - xpre;
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator();
            
            % Expected value
            xpre = coefs{1};
            for iIter = 1:nIters
                d = f(coefs{iIter+1},xpre);
                x = g(d);
                v = h(x,xpre);
                xpre = x;
            end
            coefsExpctd = v;
            
            % Actual value
            xpre = coefs{1};
            for iIter = 1:nIters
                [v,x] = testCase.target.step(...
                    @(x) h(x,xpre),...
                    @(x) g(x),...
                    @(x) f(x,xpre),...
                    coefs{iIter+1});
                xpre = x;
            end
            coefsActual = v;
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
    end
    
end