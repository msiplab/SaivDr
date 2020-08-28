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
        function cpst = softthresh(ctmp,cpre,lambda,gamma)
            u = cpre-gamma*ctmp;
            v = abs(u)-lambda;
            cpst = sign(u).*(v+abs(v))/2;
        end
        
        function cpst = coefpdshshc(ctmp,cpre,lambda,gamma)
            u = cpre-gamma*ctmp;
            v = abs(u)-lambda;
            spst = sign(u).*(v+abs(v))/2;
            cpst = 2*spst-cpre;
        end
    end
    
    methods (Test)
        
        function testDefaultStep(testCase,width,height)
            
            % Parameters
            coefstmp = randn(width,height);
            coefspre = [];
            
            % Instantiation
            import saivdr.restoration.*
            testCase.target = CoefsManipulator();
            
            % Expected value
            coefsExpctd = coefstmp;
            
            % Actual value
            coefsActual = testCase.target.step(coefstmp,coefspre);
            
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
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation',g);
            
            % Expected value
            statepre = 0;
            coefsExpctd = g(coefspre,statepre);
            
            % Actual value
            coefsActual = testCase.target.step(coefspre,statepre);
            
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
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation',g);
            
            % Expected value
            statepre = 0;            
            coefsExpctd = cell(1,nChs);
            for iCh = 1:nChs
                coefsExpctd{iCh} = g(coefspre{iCh},statepre);
            end
            
            % Actual value
            coefsActual = testCase.target.step(coefspre,statepre);
            
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
            coefstmp = randn(width,height,depth);
            
            % Function
            lambda = 1e-3;
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation',g);
            
            % Expected value
            coefspre = 0;
            coefsExpctd = g(coefstmp,coefspre);
            
            % Actual value
            coefsActual = ...
                testCase.target.step(coefstmp,coefspre);
            
            % Evaluation
            testCase.verifySize(coefsActual,size(coefsExpctd));
            diff = max(abs(coefsExpctd(:) - coefsActual(:)));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-10,...
                sprintf('%g',diff));
            
        end
        
        function testSoftThresholding3dCell(testCase,width,height,depth)
            
            % Parameters
            nChs = 5;
            coefstmp = cell(1,nChs);
            for iCh = 1:nChs
                coefstmp{iCh} = randn(width,height,depth);
            end
            
            % Function
            lambda = 1e-3;
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation',g);
            
            % Expected value
            coefspre = 0;
            coefsExpctd = cell(1,nChs);
            for iCh = 1:nChs
                coefsExpctd{iCh} = g(coefstmp{iCh},coefspre);
            end
            
            % Actual value
            coefsActual = ...
                testCase.target.step(coefstmp,coefspre);
            
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
            gamma = 1e-3;
            g = @(x,s) testCase.softthresh(x,s,lambda,gamma);
            
            % Instantiation
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            x = coefs{1};
            for iIter = 1:nIters
                x = g(coefs{iIter+1},x);
            end
            coefsExpctd = x;
            
            % Actual value
            x = coefs{1};
            for iIter = 1:nIters
                x = testCase.target.step(coefs{iIter+1},x);
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
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            s = coefs{1};
            for iIter = 1:nIters
                x = g(coefs{iIter+1},s);
            end
            coefsExpctd = x;
            
            % Actual value
            s = coefs{1};
            for iIter = 1:nIters
                x = testCase.target.step(coefs{iIter+1},s);
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
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            x = coefs{1};
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    x{iCh} = g(subcoefs{iCh},x{iCh});
                end
            end
            coefsExpctd = x;
            
            % Actual value
            x = coefs{1};
            for iIter = 1:nIters
                x = testCase.target.step(coefs{iIter+1},x);
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
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            x = coefs{1};
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    x{iCh} = g(subcoefs{iCh},x{iCh});
                end
            end
            coefsExpctd = x;
            
            % Actual value
            x = coefs{1};
            for iIter = 1:nIters
                x = testCase.target.step(coefs{iIter+1},x);
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
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            x = coefs{1};
            for iIter = 1:nIters
                x = g(coefs{iIter+1},x);
            end
            coefsExpctd = x;
            
            % Actual value
            x = coefs{1};
            for iIter = 1:nIters
                x = testCase.target.step(coefs{iIter+1},x);
            end
            coefsActual = x;
            
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
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            x = coefs{1};
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    x{iCh} = g(subcoefs{iCh},x{iCh});
                end
            end
            coefsExpctd = x;
            
            % Actual value
            x = coefs{1};
            for iIter = 1:nIters
                x = testCase.target.step(coefs{iIter+1},x);
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
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            x = coefs{1};
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    x{iCh} = g(subcoefs{iCh},x{iCh});
                end
            end
            coefsExpctd = x;
            
            % Actual value
            if verLessThan('matlab','9.4')
                x = num2cell(zeros(1,nChs));
            else
                x = 0;
            end
            for iIter = 1:nIters
                x = testCase.target.step(coefs{iIter+1},x);
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
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            x = coefs{1};
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    x{iCh} = g(subcoefs{iCh},x{iCh});
                end
            end
            coefsExpctd = x;
            
            % Actual value
            targetClone = testCase.target.clone();
            targetClone.release();
            x = coefs{1};
            for iIter = 1:nIters
                x = targetClone.step(coefs{iIter+1},x);
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
            import saivdr.restoration.*
            testCase.target = CoefsManipulator('Manipulation', g);
            
            % Expected value
            x = coefs{1};
            for iIter = 1:nIters
                subcoefs = coefs{iIter+1};
                for iCh = 1:nChs
                    x{iCh} = g(subcoefs{iCh},x{iCh});
                end
            end
            coefsExpctd = x;
            
            % Actual value
            if verLessThan('matlab','9.4')
                x = num2cell(zeros(1,nChs));
            else
                x = 0;
            end
            for iIter = 1:nIters
                x = testCase.target.step(coefs{iIter+1},x);
            end
            coefsActual = x;
            
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