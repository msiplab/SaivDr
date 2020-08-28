classdef FistaSystem < saivdr.restoration.AbstIterativeMethodSystem
    % FISTASYSTEM Signal restoration via fast iterative soft thresholding algorithm
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
    %  Fast iterative soft thresholding algorithm (FISTA)
    % -------------------------------------------------------------------
    % Input:  x(0)
    % Output: r(n)
    %  1: n = 0
    %  2: t(0) = 1
    %  3: y(0) = x(0)
    %  4: z(0) = Dy(0)
    %  5: while A stopping criterion is not satisfied do
    %  6:     u <- D'P'(Pz(n) - v)
    %  7:     x(n+1) = G_R( y(n) - gamma*u, sqrt(lambda*gamma) )
    %  8:     r(n+1) = Dx(n+1)
    %  9:     t(n+1) = (1+sqrt(1+4*t(n)^2))/2
    % 10:     a <- (t(n)-1)/t(n+1)
    % 11:     y(n+1) = x(n+1)+a*(x(n+1)-x(n))    
    % 12:     z(n+1) = r(n+1)+a*(r(n+1)-r(n))      
    % 13:     n <- n+1
    % 14: end while
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
        Y
        Z
        T
        Scales
    end

    properties(Nontunable, Hidden)
        GaussianDenoiser
    end
    
    methods
        function obj = FistaSystem(varargin)
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
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.restoration.AbstIterativeMethodSystem(...
                obj);
            s.Scales = obj.Scales;
            s.X      = obj.X;
            s.Y      = obj.Y;            
            s.Z      = obj.Z;
            s.T      = obj.T;                        
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
            obj.Y      = s.Y;            
            obj.Z      = s.Z;
            obj.T      = s.T;                        
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
                obj.Y = obj.X;
                obj.Z = obj.Result; 
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
                obj.Y = obj.X;
                obj.Z = obj.Result;                 
                obj.ParallelProcess.InitialState = obj.Y;
            end
            obj.T = 1;
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
            yPre   = obj.Y;
            zPre   = obj.Z;
            tPre   = obj.T;
            
            % Main steps
            g = adjProc.step(msrProc.step(zPre)-vObs);
            t = (1+sqrt(1+4*tPre^2))/2;
            a = (tPre-1)/t;
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
                u = adjDic.step(g);
                x = gdn.step(yPre-gamma*u);
                result = fwdDic(x,scales);
                %
                % Update
                obj.X = x;
                obj.Y = x + a*(x-xPre);                
            else % OLS/OLA process
                obj.ParallelProcess.States = yPre;
                result = obj.ParallelProcess.step(g);
                obj.X = obj.ParallelProcess.States;
                %
                nSplit = length(xPre);
                for iSplit = 1:nSplit
                    obj.Y{iSplit} = cellfun(@(u,v) u + a*(u-v),...
                        obj.X{iSplit},xPre{iSplit},...
                        'UniformOutput',false);
                end
            end
            obj.Z = result + a*(result-resPre);
            obj.T = t;
            
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