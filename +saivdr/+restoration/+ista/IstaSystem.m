classdef IstaSystem < saivdr.restoration.AbstIterativeMethodSystem
    % ISTASYSTEM Signal restoration via iterative soft thresholding algorithm
    %
    % Problem setting:
    %
    %    r^ = Dx^
    %    x^ = argmin_x (1/2)||v - PDx||_2^2 + lambda ||x||_1
    %
    % Input:
    %
    %    v : Observation
    %    P : Measurment process
    %    D : Synthesis dictionary
    %
    % Output:
    %
    %    r^: Restoration
    %
    % ===================================================================
    %  Iterative soft thresholding algorithm (ISTA)
    % -------------------------------------------------------------------
    % Input:  x(0)
    % Output: r(n)
    %  1: n = 0
    %  2: r(0) = Dx(0)
    %  3: while A stopping criterion is not satisfied do
    %  4:     t <- D'P'(Pr(n) - v)
    %  5:     x(n+1) = G_R( x(n) - gamma*t, sqrt(lambda*gamma) )
    %  6:     r(n+1) = Dx(n+1)
    %  7:     n <- n+1
    %  8: end while
    % ===================================================================
    %  G_R(x,sigma) = sign(x).*max(|x|-sigma^2,0) for R=||.||_1, and
    %  gamma = 1/L, where L is the Lipcitz constant of the gradient of the
    %  1st term.
    % -------------------------------------------------------------------
    %
    % Reference:
    %
    %
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


    properties(Access = private)
        X
        Scales
    end

    properties(Nontunable, Hidden)
        GaussianDenoiser
    end
    
    methods
        function obj = IstaSystem(varargin)
            import saivdr.restoration.AbstIterativeMethodSystem
            obj = obj@saivdr.restoration.AbstIterativeMethodSystem(...
                varargin{:});
            setProperties(obj,nargin,varargin{:})
        end
        
        function [coefs,scales] = getCoefficients(obj)
            import saivdr.restoration.AbstOlsOlaProcess
            if isa(obj.ParallelProcess,'saivdr.restoration.AbstOlsOlaProcess')
                [coefs,scales] = obj.ParallelProcess.getCoefficients();           
            else
               coefs = obj.X;
               scales = obj.Scales;
            end
        end
    end
    
    methods(Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.restoration.AbstIterativeMethodSystem(...
                obj);
            s.Scales = obj.Scales;
            s.X      = obj.X;
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
            msrProc.step(vObs);
            if isempty(obj.Gamma)
                obj.Gamma = 1/(framebound*msrProc.LambdaMax);               
            end

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
            gamma  = obj.Gamma;
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
                gdn = obj.GaussianDenoiser;
                cm = CoefsManipulator(...
                    'Manipulation',...
                    @(t,cpre) gdn.step(cpre-gamma*t));
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
        end
        
        function varargout = stepImpl(obj)
            stepImpl@saivdr.restoration.AbstIterativeMethodSystem(obj)
            % Observation
            vObs = obj.Observation;
            % Measurement process
            msrProc = obj.MeasureProcess;
            adjProc = obj.AdjointProcess;
            
            % Previous state
            resPre = obj.Result;
            xPre   = obj.X;
            
            % Main steps
            g = adjProc.step(msrProc.step(resPre)-vObs);
            if isempty(obj.SplitFactor) % Normal process
                import saivdr.restoration.denoiser.*
                % Dictionaries
                fwdDic = obj.Dictionary{obj.FORWARD};
                adjDic = obj.Dictionary{obj.ADJOINT};
                scales = obj.Scales;
                %
                gamma  = obj.Gamma;
                gdn = obj.GaussianDenoiser;
                %
                t = adjDic.step(g);
                x = gdn.step(xPre-gamma*t);
                result = fwdDic(x,scales);
                % Update
                obj.X = x;
            else % OLS/OLA process
                result = obj.ParallelProcess.step(g);
            end
            
            % Output
            if nargout > 0
                varargout{1} = result;
            end
            if nargout > 1
                import saivdr.restoration.AbstIterativeMethodSystem
                varargout{2} = AbstIterativeMethodSystem.rmse(result,resPre);
            end
            
            % Update
            obj.Result = result;
        end        
        
    end
           
end