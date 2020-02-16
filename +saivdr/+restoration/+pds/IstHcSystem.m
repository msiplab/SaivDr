classdef IstHcSystem < saivdr.restoration.AbstIterativeMethodSystem
    % ISTHCSYSTEM Signal restoration via primal-dual splitting method
    %
    % Problem setting:
    %
    %    r^ = Dx^
    %    x^ = argmin_x (1/2)||v - PDx||_2^2 + lambda ||x||_1,
    %         s.t. Dx in C
    %
    % Input:
    %
    %    v : Observation
    %    P : Measurment process
    %    D : Synthesis dictionary
    %    C : Constraint s.t. the prox is available
    %
    % Output:
    %
    %    r^: Restoration
    %
    % ===================================================================
    %  Iterative soft thresholding with hard constraint 
    % -------------------------------------------------------------------
    % Input:  x(0), y(0)
    % Output: r(n)
    %  1: n = 0
    %  2: r(0) = Dx(0)
    %  3: while A stopping criterion is not satisfied do
    %  4:     t <- D'(P'(Pr(n) - v) + y(n))
    %  5:     x(n+1) = G_R( x(n) - gamma1*t, sqrt(lambda*gamma1) )
    %  6:     r(n+1) = 2Dx(n+1)
    %  7:     u(n) <-  2r(n+1) - r(n)
    %  8:     y(n) <-  y(n) + gamma2*u(n)
    %  9:     y(n+1) = y(n) - gamma2*Pc(y(n)/gamma2)
    % 10:     n <- n+1
    % 11: end while
    % ===================================================================
    %  G_R(x,sigma) = sign(x).*max(|x|-sigma^2,0) for R=||.||_1
    % -------------------------------------------------------------------
    %
    % Reference:
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
    
    properties(Nontunable, Hidden)
        GaussianDenoiser
    end    
    
    properties(Nontunable)
        MetricProjection
    end
    
    properties(Access = private)
        X
        Y
        Scales
    end
    
    methods
        function obj = IstHcSystem(varargin)
            import saivdr.restoration.AbstIterativeMethodSystem
            obj = obj@saivdr.restoration.AbstIterativeMethodSystem(...
                varargin{:});
            setProperties(obj,nargin,varargin{:})
        end
        
        function [coefs,scales] = getCoefficients(obj)
            if isempty(obj.ParallelProcess)
               coefs = obj.X;
               scales = obj.Scales;
            else
                [coefs,scales] = obj.ParallelProcess.getCoefficients();           
            end
        end
    end
    
    methods(Access = protected)
        
        function validatePropertiesImpl(obj)
            if isempty(obj.MetricProjection)
                error('MetricProjection must be set.')
            end
        end        
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.restoration.AbstIterativeMethodSystem(...
                obj);
            s.Scales = obj.Scales;
            s.X      = obj.X;
            s.Y      = obj.Y;
            %s.Var = obj.Var;
            %s.Obj = matlab.System.saveObject(obj.Obj);
            %if isLocked(obj)
            %    s.Iteration = obj.Iteration;
            %end
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            %if wasLocked
            %    obj.Iteration = s.Iteration;
            %end
            %obj.Obj = matlab.System.loadObject(s.Obj);
            %obj.Var = s.Var;
            obj.Y      = s.Y;
            obj.X      = s.X;
            obj.Scales = s.Scales;
            loadObjectImpl@saivdr.restoration.AbstIterativeMethodSystem(...
                obj,s,wasLocked);
        end
        
        function setupImpl(obj)
            setupImpl@saivdr.restoration.AbstIterativeMethodSystem(obj);
            % Observation
            vObs = obj.Observation;
            % Dictionary
            fwdDic = obj.Dictionary{obj.FORWARD};
            adjDic = obj.Dictionary{obj.ADJOINT};
            % Measurement process
            msrProc = obj.MeasureProcess;
            
            % Calculation of step size parameter
            framebound = fwdDic.FrameBound;
            %msrProc.step(vObs);
            if isempty(obj.Gamma)
                obj.Gamma = cell(1,2);
                obj.Gamma{1} = 1/(framebound*msrProc.LambdaMax);
            end
            obj.Gamma{2} = 1/(1.05*obj.Gamma{1}*framebound);            
            
            % Adjoint of measuremnt process
            adjProc = msrProc.clone();
            adjProc.release();
            adjProc.ProcessingMode = 'Adjoint';
            obj.AdjointProcess = adjProc;
            
            % Gaussian denoiser
            if isempty(obj.GaussianDenoiser)
                import saivdr.restoration.denoiser.*
                obj.GaussianDenoiser = GaussianDenoiserSfth();
            end
            gamma  = obj.Gamma{1};
            lambda = obj.Lambda;
            obj.GaussianDenoiser.Sigma = sqrt(gamma*lambda);            
            
            % Initialization
            obj.Result = adjProc.step(zeros(size(vObs),'like',vObs));
            if isempty(obj.SplitFactor)
                obj.ParallelProcess = [];
                %
                fwdDic.release();
                obj.Dictionary{obj.FORWARD} = fwdDic.clone();
                adjDic.release();
                obj.Dictionary{obj.ADJOINT} = adjDic.clone();
                %
                [obj.X,obj.Scales] = adjDic.step(obj.Result);
            else
                import saivdr.restoration.*
                import saivdr.restoration.denoiser.*
                gamma1  = obj.Gamma{1};
                gdn = obj.GaussianDenoiser;
                cm = CoefsManipulator(...
                    'Manipulation',...
                    @(t,cpre) gdn.step(cpre-gamma1*t));
                if strcmp(obj.DataType,'Volumetric Data')
                    obj.ParallelProcess = OlsOlaProcess3d();
                else
                    obj.ParallelProcess = OlsOlaProcess2d();
                end
                obj.ParallelProcess.Synthesizer = fwdDic;
                obj.ParallelProcess.Analyzer    = adjDic;
                obj.ParallelProcess.CoefsManipulator = cm;
                obj.ParallelProcess.SplitFactor = obj.SplitFactor;
                obj.ParallelProcess.PadSize     = obj.PadSize;
                obj.ParallelProcess.UseParallel = obj.UseParallel;
                obj.ParallelProcess.UseGpu      = obj.UseGpu;
                obj.ParallelProcess.IsIntegrityTest = obj.IsIntegrityTest;
                obj.ParallelProcess.Debug       = obj.Debug;
                %
                obj.X = obj.ParallelProcess.analyze(obj.Result);
                obj.ParallelProcess.InitialState = obj.X;
            end
            obj.Y = zeros(size(vObs),'like',vObs);
        end
        
        function varargout = stepImpl(obj)
            stepImpl@saivdr.restoration.AbstIterativeMethodSystem(obj)
            % Observation
            vObs = obj.Observation;
            % Preparation
            msrProc = obj.MeasureProcess;
            mtrProj = obj.MetricProjection;
            adjProc = obj.AdjointProcess;
            
            % Previous state
            resPre = obj.Result;
            xPre   = obj.X;
            yPre   = obj.Y;
            % Main steps
            g = adjProc.step(msrProc.step(resPre)-vObs)+yPre;
            if isempty(obj.SplitFactor) % Normal process
                import saivdr.restoration.denoiser.*
                % Dictionaries
                fwdDic = obj.Dictionary{obj.FORWARD};
                adjDic = obj.Dictionary{obj.ADJOINT};
                scales = obj.Scales;
                %
                gamma1  = obj.Gamma{1};
                gdn = obj.GaussianDenoiser;
                %
                t = adjDic.step(g);
                x = gdn.step(xPre-gamma1*t);
                result = fwdDic(x,scales);
                % Update
                obj.X = x;
            else % OLS/OLA process
                result = obj.ParallelProcess.step(g);
            end
            u = 2*result - resPre;
            gamma2 = obj.Gamma{2};
            y = yPre + gamma2*u;
            y = y - gamma2*mtrProj.step(y/gamma2);

            % Output
            if nargout > 0
                varargout{1} = result;
            end
            if nargout > 1
                import saivdr.restoration.AbstIterativeMethodSystem
                varargout{2} = AbstIterativeMethodSystem.rmse(result,resPre);
            end
            
            % Update
            obj.Y = y;
            obj.Result = result;
        end
        
    end
    
end