classdef ProxBoxConstraintTestCase < matlab.unittest.TestCase
    %PROXBOXCONSTRAINT Test caes for ProxBoxConstraint
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
        vmin = struct('neginf', -Inf,'negone', -1,'zero', 0 )
        vmax = struct('zero', 0,'posone', 1, 'posinf', Inf )
        width = struct('small',1,'medium',8,'large',16)
        height = struct('small',1,'medium',8,'large',16)
        depth = struct('small',1,'medium',8,'large',16)
        sigma = struct('small',1e-1,'medium',1e0,'large',1e1)
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
            
            vminExpctd = -Inf;
            vmaxExpctd = Inf;
            
            import saivdr.restoration.metricproj.*
            testCase.target = ProxBoxConstraint();
            
            vminActual = testCase.target.Vmin;
            vmaxActual = testCase.target.Vmax;
            
            testCase.verifyEqual(vminActual,vminExpctd);
            testCase.verifyEqual(vmaxActual,vmaxExpctd);            
            
        end
        
        
        function testVminVmax(testCase,vmin,vmax)
            
            % Expected values
            vminExpctd = vmin;
            vmaxExpctd = vmax;
            
            % Instantiation
            import saivdr.restoration.metricproj.*
            testCase.target = ProxBoxConstraint(...
                'Vmin',vmin,...
                'Vmax',vmax);
            
            % Actual values            
            vminActual = testCase.target.Vmin;
            vmaxActual = testCase.target.Vmax;
            
            % Evaluation
            testCase.verifyEqual(vminActual,vminExpctd);
            testCase.verifyEqual(vmaxActual,vmaxExpctd);            
            
        end        

        function testStepScalar(testCase,...
                vmin,vmax,width,height,depth,sigma)
            
            % Parameters
            src = sigma*randn(height,width,depth);
            
            % Expected values
            vminExpctd = vmin;
            vmaxExpctd = vmax;
            resExpctd = src;
            resExpctd(resExpctd<=vmin) = vmin;
            resExpctd(resExpctd>=vmax) = vmax;
            
            % Instantiation
            import saivdr.restoration.metricproj.*
            testCase.target = ProxBoxConstraint(...
                'Vmin',vmin,...
                'Vmax',vmax);
            
            % Actual values
            resActual = testCase.target.step(src);
            vminActual = min(resActual(:));
            vmaxActual = max(resActual(:));
            
            % Evaluation
            eps = 1e-10;
            import matlab.unittest.constraints.IsLessThanOrEqualTo
            import matlab.unittest.constraints.IsGreaterThanOrEqualTo
            testCase.verifyThat(vminActual,IsGreaterThanOrEqualTo(vminExpctd));
            testCase.verifyThat(vmaxActual,IsLessThanOrEqualTo(vmaxExpctd));
            diff = max(abs(resExpctd(:)-resActual(:)));
            testCase.verifyEqual(resActual,resExpctd,...
                'AbsTol',eps,num2str(diff));
        end
        
        %{
        function testStepVector(testCase,inputSize,sigma)
            
            x    = sigma*randn(inputSize,1);
            svec = sigma*rand(inputSize,1);
            
            v = abs(x)-svec.^2;
            v(v<0) = 0;
            yExpctd = sign(x).*v;
            
            target = PlgGdnSfth('Sigma',svec);
            
            yActual = target.step(x);
            
            testCase.verifyEqual(yActual,yExpctd);
            
        end
        %}
    end
    
end

