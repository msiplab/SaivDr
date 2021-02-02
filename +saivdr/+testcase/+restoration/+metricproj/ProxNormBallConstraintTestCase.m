classdef ProxNormBallConstraintTestCase < matlab.unittest.TestCase
    %TESTCASEPLGSOFTTHRESHOLDING Test caes for ProxNormBallConstraint
    %
    % Requirements: MATLAB R2018a
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
        eps = struct('small',1e-2,'medium',1e-1, 'large',1e0')
        width = struct('small',1,'medium',8,'large',16)
        height = struct('small',1,'medium',8,'large',16)
        depth = struct('small',1,'medium',8,'large',16)
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
        
        function testConstruction(testCase)
            
            epsExpctd = Inf;
            centerExpctd = 0;
            
            import saivdr.restoration.metricproj.*
            testCase.target = ProxNormBallConstraint();
            
            epsActual = testCase.target.Eps;
            centerActual = testCase.target.Center;
            
            testCase.verifyEqual(epsActual,epsExpctd);
            testCase.verifyEqual(centerActual,centerExpctd);            
            
        end
    
        function testEps(testCase,...
                height,width,depth,eps)
            
            % Expected valuses
            epsExpctd = eps;
            centerExpctd = randn(height,width,depth);
            
            % Instantiation
            import saivdr.restoration.metricproj.*
            testCase.target = ProxNormBallConstraint(...
                'Eps',eps);
            testCase.target.Center = centerExpctd;
            
            % Actual values
            epsActual = testCase.target.Eps;
            centerActual = testCase.target.Center;
            
            % Evaluation
            testCase.verifyEqual(epsActual,epsExpctd);
            testCase.verifyEqual(centerActual,centerExpctd);            
            
        end
           
        function testStep(testCase,...
                height,width,depth,eps)
            
            % Parameters
            src    = randn(height,width,depth);
            center = randn(height,width,depth);
            
            % Expected valuses
            if norm(src(:)-center(:),2)<=eps
                resExpctd = src;
            else
                resExpctd = center + ...
                    (eps/norm(src(:)-center(:),2))*(src-center);
            end
            
            % Instantiation
            import saivdr.restoration.metricproj.*
            testCase.target = ProxNormBallConstraint(...
                'Eps',eps);
            testCase.target.Center = center;
            
            % Actual values
            resActual = testCase.target.step(src);
            
            % Evaluation
            epsEv = 1e-10;
            diff = max(abs(resExpctd(:)-resActual(:)));
            testCase.verifyEqual(resActual,resExpctd,...
                'AbsTol',epsEv,num2str(diff));
            
            % Change center
            center = center + randn(height,width,depth);
            testCase.target.Center = center;
            if norm(src(:)-center(:),2)<=eps
                resExpctd = src;
            else
                resExpctd = center + ...
                    (eps/norm(src(:)-center(:),2))*(src-center);
            end            
            
            % Actual values
            resActual = testCase.target.step(src);
            
            % Evaluation
            epsEv = 1e-10;
            diff = max(abs(resExpctd(:)-resActual(:)));
            testCase.verifyEqual(resActual,resExpctd,...
                'AbsTol',epsEv,num2str(diff));            

        end            
        
    end
    
end

