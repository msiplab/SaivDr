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
    
    methods (Static)
        function y = softthresh(x,xpre,lambda,gamma)
            u = xpre-gamma*x;
            v = abs(u)-lambda;
            y = sign(u).*(v+abs(v))/2;
        end
        
        function [v,x] = coefpdshshc(t,xpre,lambda,gamma)
            u = xpre-gamma*t;
            w = abs(u)-lambda;
            x = sign(u).*(w+abs(w))/2;
            v = 2*x - xpre;
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
        
        function testClone(testCase,width,height)
            
            % Parameters
            stateExpctd = randn(width,height);
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = CoefsManipulator('InitialState',stateExpctd);
            
            % Clone
            cloneCoefsManipulator = clone(testCase.target);
            stateActual = cloneCoefsManipulator.InitialState;
            
            % Evaluation
            testCase.verifyEqual(stateActual,stateExpctd);
            
            % Check independency
            stateNotExpctd = randn(width,height);
            testCase.target.InitialState = stateNotExpctd;
            stateActual = cloneCoefsManipulator.InitialState;
            
            % Evaluation
            testCase.verifyNotEqual(stateActual,stateNotExpctd);
            
        end
        
        
        function testCloneCellInitialState(testCase,width,height)
            
            % Parameters
            nCells = 2;
            stateExpctd = cell(nCells,1);
            for idx = 1:nCells
                stateExpctd{idx} = randn(width,height);
            end
            
            % Instantiation of target class
            import saivdr.utility.*
            testCase.target = CoefsManipulator('InitialState',stateExpctd);
            
            % Clone
            cloneCoefsManipulator = clone(testCase.target);
            stateActual = cloneCoefsManipulator.InitialState;
            
            % Evaluation
            for idx = 1:nCells
                testCase.verifyEqual(stateActual{idx},stateExpctd{idx});
            end
            
            % Check independency
            stateNotExpctd = cell(nCells,1);
            for idx = 1:nCells
                stateNotExpctd{idx} = randn(width,height);
            end
            testCase.target.InitialState = stateNotExpctd;
            stateActual = cloneCoefsManipulator.InitialState;
            
            % Evaluation
            testCase.verifyNotEqual(stateActual,stateNotExpctd);
            
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
            testCase.target.Manipulation = g;
            coefsActual = testCase.target.step(coefspre);
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
        
        function testSoftThresholding2dCell(testCase,width,height)
            
            % Parameters
            nChs = 5;
            coefspre = cell(1,nChs);
            for iCh = 1:nChs
                coefspre{iCh} = randn(width,height);
            end
            
            % Function
            lambda = 1e-3;
            g = @(x) sign(x).*((abs(x)-lambda)+abs(abs(x)-lambda))/2;
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator();
            
            % Expected value
            coefsExpctd = cell(1,nChs);
            for iCh = 1:nChs
                coefsExpctd{iCh} = g(coefspre{iCh});
            end
            
            % Actual value
            testCase.target.Manipulation = g;
            coefsActual = testCase.target.step(coefspre);
            
            % Evaluation
            for iCh = 1:nChs
                testCase.verifySize(coefsActual{iCh},size(coefsExpctd{iCh}));
                diff = max(abs(coefsExpctd{iCh}(:) - coefsActual{iCh}(:)));
                testCase.verifyEqual(coefsActual{iCh},coefsExpctd{iCh},...
                    'AbsTol',1e-10,sprintf('%g',diff));
            end
            
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
            testCase.target.Manipulation = g;
            coefsActual = testCase.target.step(coefspre);
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
        
        function testSoftThresholding3dCell(testCase,width,height,depth)
            
            % Parameters
            nChs = 5;
            coefspre = cell(1,nChs);
            for iCh = 1:nChs
                coefspre{iCh} = randn(width,height,depth);
            end
            
            % Function
            lambda = 1e-3;
            g = @(x) sign(x).*((abs(x)-lambda)+abs(abs(x)-lambda))/2;
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator();
            
            % Expected value
            coefsExpctd = cell(1,nChs);
            for iCh = 1:nChs
                coefsExpctd{iCh} = g(coefspre{iCh});
            end
            
            % Actual value
            testCase.target.Manipulation = g;
            coefsActual = testCase.target.step(coefspre);
            
            % Evaluation
            for iCh = 1:nChs
                testCase.verifySize(coefsActual{iCh},size(coefsExpctd{iCh}));
                diff = max(abs(coefsExpctd{iCh}(:) - coefsActual{iCh}(:)));
                testCase.verifyEqual(coefsActual{iCh},coefsExpctd{iCh},...
                    'AbsTol',1e-10,sprintf('%g',diff));
            end
            
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
            f = @(x,xpre) testCase.softthresh(x,xpre,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator(...
                'Manipulation', f, ...
                'IsFeedBack',true);
            
            % Expected value
            xpre = coefs{1};
            for iIter = 1:nIters
                xpre = f(coefs{iIter+1},xpre);
            end
            coefsExpctd = xpre;
            
            % Actual value
            testCase.target.InitialState = coefs{1};
            for iIter = 1:nIters
                x = testCase.target.step(coefs{iIter+1});
            end
            coefsActual = x;
            
            % Evaluation
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
            f = @(x,xpre) testCase.softthresh(x,xpre,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator(...
                'Manipulation', f, ...
                'IsFeedBack',true);
            
            % Expected value
            xpre = coefs{1};
            for iIter = 1:nIters
                xpre = f(coefs{iIter+1},xpre);
            end
            coefsExpctd = xpre;
            
            % Actual value
            testCase.target.InitialState = coefs{1};
            for iIter = 1:nIters
                x = testCase.target.step(coefs{iIter+1});
            end
            coefsActual = x;
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
        
        function testIterativeSoftThresholding2dCell(testCase,width,height)
            
            % Parameters
            nIters = 5;
            nChs = 5;
            coefs  = cell(nIters+1,1);
            for iIter = 1:nIters+1
                subcoefs = cell(1,nChs);
                for iCh = 1:nChs
                    subcoefs{iCh} = randn(width,height);
                end
                coefs{iIter} = subcoefs;
            end
            
            % Function
            lambda = 1e-3;
            gamma  = 1e-3;
            f = @(x,xpre) testCase.softthresh(x,xpre,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator(...
                'Manipulation', f, ...
                'IsFeedBack',true);
            
            % Expected value
            xpre = coefs{1};
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    xpre{iCh} = f(subcoefs{iCh},xpre{iCh});
                end
            end
            coefsExpctd = xpre;
            
            % Actual value
            testCase.target.InitialState = coefs{1};
            for iIter = 1:nIters
                x = testCase.target.step(coefs{iIter+1});
            end
            coefsActual = x;
            
            % Evaluation
            for iCh = 1:nChs
                testCase.verifySize(coefsActual{iCh},size(coefsExpctd{iCh}));
                diff = max(abs(coefsExpctd{iCh}(:) - coefsActual{iCh}(:)));
                testCase.verifyEqual(coefsActual{iCh},coefsExpctd{iCh},...
                    'AbsTol',1e-10,sprintf('%g',diff));
            end
        end
        
        function testIterativeSoftThresholding3dCell(testCase,width,height,depth)
            
            % Parameters
            nIters = 5;
            nChs = 5;
            coefs  = cell(nIters+1,1);
            for iIter = 1:nIters+1
                subcoefs = cell(1,nChs);
                for iCh = 1:nChs
                    subcoefs{iCh} = randn(width,height,depth);
                end
                coefs{iIter} = subcoefs;
            end
            
            % Function
            lambda = 1e-3;
            gamma  = 1e-3;
            f = @(x,xpre) testCase.softthresh(x,xpre,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator(...
                'Manipulation', f, ...
                'IsFeedBack',true);
            
            % Expected value
            xpre = coefs{1};
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    xpre{iCh} = f(subcoefs{iCh},xpre{iCh});
                end
            end
            coefsExpctd = xpre;
            
            % Actual value
            testCase.target.InitialState = coefs{1};
            for iIter = 1:nIters
                x = testCase.target.step(coefs{iIter+1});
            end
            coefsActual = x;
            
            % Evaluation
            for iCh = 1:nChs
                testCase.verifySize(coefsActual{iCh},size(coefsExpctd{iCh}));
                diff = max(abs(coefsExpctd{iCh}(:) - coefsActual{iCh}(:)));
                testCase.verifyEqual(coefsActual{iCh},coefsExpctd{iCh},...
                    'AbsTol',1e-10,sprintf('%g',diff));
            end
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
            f = @(x,xpre) testCase.coefpdshshc(x,xpre,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator(...
                'Manipulation', f, ...
                'IsFeedBack',true,...
                'IsStateOutput',true);
            
            % Expected value
            xpre = coefs{1};
            for iIter = 1:nIters
                [v,xpre] = f(coefs{iIter+1},xpre);
            end
            coefsExpctd = v;
            
            % Actual value
            testCase.target.InitialState = coefs{1};
            for iIter = 1:nIters
                v = testCase.target.step(coefs{iIter+1});
            end
            coefsActual = v;
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
        
        function testPdsHsHcOct3dCell(testCase,width,height,depth)
            
            % Parameters
            nIters = 5;
            nChs = 5;
            coefs  = cell(nIters+1,1);
            for iIter = 1:nIters+1
                subcoefs = cell(1,nChs);
                for iCh = 1:nChs
                    subcoefs{iCh} = randn(width,height,depth);
                end
                coefs{iIter} = subcoefs;
            end
            
            % Function
            lambda = 1e-3;
            gamma  = 1e-3;
            f = @(x,xpre) testCase.coefpdshshc(x,xpre,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator(...
                'Manipulation', f, ...
                'IsFeedBack',true,...
                'IsStateOutput',true);
            
            % Expected value
            xpre = coefs{1};
            v = cell(1,nChs);
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    [v{iCh},xpre{iCh}] = f(subcoefs{iCh},xpre{iCh});
                end
            end
            coefsExpctd = v;
            
            % Actual value
            testCase.target.InitialState = coefs{1};
            for iIter = 1:nIters
                v = testCase.target.step(coefs{iIter+1});
            end
            coefsActual = v;
            
            % Evaluation
            for iCh = 1:nChs
                testCase.verifySize(coefsActual{iCh},size(coefsExpctd{iCh}));
                diff = max(abs(coefsExpctd{iCh}(:) - coefsActual{iCh}(:)));
                testCase.verifyEqual(coefsActual{iCh},coefsExpctd{iCh},...
                    'AbsTol',1e-10,sprintf('%g',diff));
            end
        end
        
        function testPdsHsHcOct3dCellWithValueState(testCase,width,height,depth)
            
            % Parameters
            nIters = 5;
            nChs = 5;
            coefs  = cell(nIters+1,1);
            for iIter = 1:nIters+1
                subcoefs = cell(1,nChs);
                for iCh = 1:nChs
                    if iIter == 1
                        subcoefs{iCh} = zeros(width,height,depth);
                    else
                        subcoefs{iCh} = randn(width,height,depth);                        
                    end
                end
                coefs{iIter} = subcoefs;
            end
            
            % Function
            lambda = 1e-3;
            gamma  = 1e-3;
            f = @(x,xpre) testCase.coefpdshshc(x,xpre,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator(...
                'Manipulation', f, ...
                'IsFeedBack',true,...
                'IsStateOutput',true);
            
            % Expected value
            xpre = coefs{1};
            v = cell(1,nChs);
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    [v{iCh},xpre{iCh}] = f(subcoefs{iCh},xpre{iCh});
                end
            end
            coefsExpctd = v;
            
            % Actual value
            testCase.target.InitialState = 0;
            for iIter = 1:nIters
                v = testCase.target.step(coefs{iIter+1});
            end
            coefsActual = v;
            
            % Evaluation
            for iCh = 1:nChs
                testCase.verifySize(coefsActual{iCh},size(coefsExpctd{iCh}));
                diff = max(abs(coefsExpctd{iCh}(:) - coefsActual{iCh}(:)));
                testCase.verifyEqual(coefsActual{iCh},coefsExpctd{iCh},...
                    'AbsTol',1e-10,sprintf('%g',diff));
            end
        end
        
        
        function testIteretiveStepsCloneCell(testCase,width,height)
            
            % Parameters
            nIters = 5;
            nChs = 5;
            coefs  = cell(nIters+1,1);
            for iIter = 1:nIters+1
                subcoefs = cell(1,nChs);
                for iCh = 1:nChs
                    subcoefs{iCh} = randn(width,height);
                end
                coefs{iIter} = subcoefs;
            end
            
            % Function
            lambda = 1e-3;
            gamma  = 1e-3;
            f = @(x,xpre) testCase.softthresh(x,xpre,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator(...
                'Manipulation', f, ...
                'IsFeedBack',true);
            
            % Expected value
            xpre = coefs{1};
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    xpre{iCh} = f(subcoefs{iCh},xpre{iCh});
                end
            end
            coefsExpctd = xpre;
            
            % Actual value
            targetClone = testCase.target.clone();
            targetClone.release();
            targetClone.InitialState = coefs{1};
            for iIter = 1:nIters
                x = targetClone.step(coefs{iIter+1});
            end
            coefsActual = x;
            
            % Evaluation
            for iCh = 1:nChs
                testCase.verifySize(coefsActual{iCh},size(coefsExpctd{iCh}));
                diff = max(abs(coefsExpctd{iCh}(:) - coefsActual{iCh}(:)));
                testCase.verifyEqual(coefsActual{iCh},coefsExpctd{iCh},...
                    'AbsTol',1e-10,sprintf('%g',diff));
            end
        end
        
    end
    
end