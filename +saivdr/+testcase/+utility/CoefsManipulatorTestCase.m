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
        usegpu = struct('true', true, 'false', false);
        dtype = { 'double', 'single' };
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
        function [y,spst] = softthresh(x,spre,lambda,gamma)
            u = spre-gamma*x;
            v = abs(u)-lambda;
            spst = sign(u).*(v+abs(v))/2;
            y = spst;
        end
        
        function [y,spst] = coefpdshshc(x,spre,lambda,gamma)
            u = spre-gamma*x;
            v = abs(u)-lambda;
            spst = sign(u).*(v+abs(v))/2;
            y = 2*spst-spre;
        end
    end
    
    methods (Test)
        
        function testDefaultStep(testCase,width,height)
            
            % Parameters
            coefspre = randn(width,height);
            statepre = [];
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator();
            
            % Expected value
            coefsExpctd = coefspre;
            stateExpctd = statepre;
            
            % Actual value
            [coefsActual,stateActual] = testCase.target.step(coefspre,statepre);
            
            % Evaluation
            testCase.verifySize(stateActual,size(stateExpctd));
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
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation',g);
            
            % Expected value
            statepre = 0;
            [coefsExpctd,stateExpctd] = g(coefspre,statepre);
            
            % Actual value
            [coefsActual,stateActual] = testCase.target.step(coefspre,statepre);
            
            % Evaluation
            testCase.verifySize(stateActual,size(stateExpctd));            
            diff = max(abs(stateExpctd(:) - stateActual(:)));
            testCase.verifyEqual(stateActual,stateExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));            
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
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation',g);
            
            % Expected value
            statepre = 0;            
            coefsExpctd = cell(1,nChs);
            stateExpctd = cell(1,nChs);
            for iCh = 1:nChs
                [coefsExpctd{iCh},stateExpctd{iCh}] = ...
                    g(coefspre{iCh},statepre);
            end
            
            % Actual value
            [coefsActual,stateActual] = ...
                testCase.target.step(coefspre,statepre);
            
            % Evaluation
            for iCh = 1:nChs
                testCase.verifySize(stateActual{iCh},size(stateExpctd{iCh}));
                diff = max(abs(stateExpctd{iCh}(:) - stateActual{iCh}(:)));
                testCase.verifyEqual(stateActual{iCh},stateExpctd{iCh},...
                    'AbsTol',1e-10,sprintf('%g',diff));                
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
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation',g);
            
            % Expected value
            statepre = 0;
            [coefsExpctd,stateExpctd] = g(coefspre,statepre);
            
            % Actual value
            [coefsActual,stateActual] = ...
                testCase.target.step(coefspre,statepre);
            
            % Evaluation
            testCase.verifySize(stateActual,size(stateExpctd));
            diff = max(abs(stateExpctd(:) - stateActual(:)));
            testCase.verifyEqual(stateActual,stateExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff))            
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
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation',g);
            
            % Expected value
            statepre = 0;
            stateExpctd = cell(1,nChs);
            coefsExpctd = cell(1,nChs);
            for iCh = 1:nChs
                [coefsExpctd{iCh},stateExpctd{iCh}] = ...
                    g(coefspre{iCh},statepre);
            end
            
            % Actual value
            [coefsActual,stateActual] = ...
                testCase.target.step(coefspre,statepre);
            
            % Evaluation
            for iCh = 1:nChs
                testCase.verifySize(stateActual{iCh},size(stateExpctd{iCh}));
                diff = max(abs(stateExpctd{iCh}(:) - stateActual{iCh}(:)));
                testCase.verifyEqual(stateActual{iCh},stateExpctd{iCh},...
                    'AbsTol',1e-10,sprintf('%g',diff));                
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
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            s = coefs{1};
            for iIter = 1:nIters
                [x,s] = g(coefs{iIter+1},s);
            end
            coefsExpctd = x;
            
            % Actual value
            s = coefs{1};
            for iIter = 1:nIters
                [x,s] = testCase.target.step(coefs{iIter+1},s);
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
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            s = coefs{1};
            for iIter = 1:nIters
                [x,s] = g(coefs{iIter+1},s);
            end
            coefsExpctd = x;
            
            % Actual value
            s = coefs{1};
            for iIter = 1:nIters
                [x,s] = testCase.target.step(coefs{iIter+1},s);
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
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            s = coefs{1};
            x = cell(1,nChs);
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    [x{iCh},s{iCh}] = g(subcoefs{iCh},s{iCh});
                end
            end
            coefsExpctd = x;
            
            % Actual value
            s = coefs{1};
            for iIter = 1:nIters
                [x,s] = testCase.target.step(coefs{iIter+1},s);
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
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            s = coefs{1};
            x = cell(1,nChs);
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    [x{iCh},s{iCh}] = g(subcoefs{iCh},s{iCh});
                end
            end
            coefsExpctd = x;
            
            % Actual value
            s = coefs{1};
            for iIter = 1:nIters
                [x,s] = testCase.target.step(coefs{iIter+1},s);
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
            g = @(x,s) testCase.coefpdshshc(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            s = coefs{1};
            for iIter = 1:nIters
                [v,s] = g(coefs{iIter+1},s);
            end
            coefsExpctd = v;
            
            % Actual value
            s = coefs{1};
            for iIter = 1:nIters
                [v,s] = testCase.target.step(coefs{iIter+1},s);
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
            g = @(x,s) testCase.coefpdshshc(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            s= coefs{1};
            v = cell(1,nChs);
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    [v{iCh},s{iCh}] = g(subcoefs{iCh},s{iCh});
                end
            end
            coefsExpctd = v;
            
            % Actual value
            s = coefs{1};
            for iIter = 1:nIters
                [v,s] = testCase.target.step(coefs{iIter+1},s);
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
            g = @(x,s) testCase.coefpdshshc(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            s = coefs{1};
            v = cell(1,nChs);
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    [v{iCh},s{iCh}] = g(subcoefs{iCh},s{iCh});
                end
            end
            coefsExpctd = v;
            
            % Actual value
            if verLessThan('matlab','9.4')
                s = num2cell(zeros(1,nChs));
            else
                s = 0;
            end
            for iIter = 1:nIters
                [v,s] = testCase.target.step(coefs{iIter+1},s);
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
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            s = coefs{1};
            x = cell(1,nChs);
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    [x{iCh},s{iCh}] = g(subcoefs{iCh},s{iCh});
                end
            end
            coefsExpctd = x;
            
            % Actual value
            targetClone = testCase.target.clone();
            targetClone.release();
            s = coefs{1};
            for iIter = 1:nIters
                [x,s] = targetClone.step(coefs{iIter+1},s);
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
        
        function testPdsHsHcOct3dCellWithValueStateDataType(testCase,...
                dtype,usegpu)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            
            % Parameters
            height_ = 32;
            width_ = 32;
            depth_ = 32;
            nIters = 5;
            nChs = 5;
            coefs  = cell(nIters+1,1);
            for iIter = 1:nIters+1
                subcoefs = cell(1,nChs);
                for iCh = 1:nChs
                    if iIter == 1
                        subcoefs{iCh} = zeros(width_,height_,depth_,dtype);
                    else
                        subcoefs{iCh} = randn(width_,height_,depth_,dtype);                        
                    end
                    if usegpu
                        subcoefs{iCh} = gpuArray(subcoefs{iCh});
                    end
                end
                coefs{iIter} = subcoefs;
            end
            
            % Function
            lambda = 1e-3;
            gamma  = 1e-3;
            g = @(x,s) testCase.coefpdshshc(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.utility.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            s = coefs{1};
            v = cell(1,nChs);
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    [v{iCh},s{iCh}] = g(subcoefs{iCh},s{iCh});
                end
            end
            coefsExpctd = v;
            
            % Actual value
            if verLessThan('matlab','9.4')
                s = num2cell(zeros(1,nChs));
            else
                s = 0;
            end
            for iIter = 1:nIters
                [v,s] = testCase.target.step(coefs{iIter+1},s);
            end
            coefsActual = v;
            
            % Evaluation
            if strcmp(dtype,'double')
                tol = 1e-10;
            else
                tol = single(1e-8);
            end
            for iCh = 1:nChs
                if usegpu
                    testCase.verifyClass(coefsActual{iCh},'gpuArray');
                    coefsActual{iCh} = gather(coefsActual{iCh});
                    coefsExpctd{iCh} = gather(coefsExpctd{iCh});
                end
                testCase.verifyClass(coefsActual{iCh},dtype);
                testCase.verifySize(coefsActual{iCh},size(coefsExpctd{iCh}));
                diff = max(abs(coefsExpctd{iCh}(:) - coefsActual{iCh}(:)));
                testCase.verifyEqual(coefsActual{iCh},coefsExpctd{iCh},...
                    'AbsTol',tol,sprintf('%g',diff));
            end
        end
                

    end
    
end