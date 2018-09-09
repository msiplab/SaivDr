classdef IstaSystem < saivdr.restoration.AbstIterativeMethodSystem
    % ISTASYSTEM Signal restoration via iterative soft thresholding algorithm
    %
    % Problem setting:
    %
    %    r^ = Dx^
    %    x^ = argmin_x (1/2)||y - PDx||_2^2 + lambda ||x||_1
    %
    % Input:
    %
    %    y : Observation
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
    %  4:     t <- D'P'(Pr(n) - y)
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
    
    methods
        function obj = IstaSystem(varargin)
            import saivdr.restoration.AbstIterativeMethodSystem
            obj = obj@saivdr.restoration.AbstIterativeMethodSystem(...
                varargin{:});
            setProperties(obj,nargin,varargin{:})
        end
    end
    
    properties(Nontunable, Access = private)
        AdjointProcess
        %parProc
    end

    properties(Access = private)
        X
        Scales
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
            % Dictionarie
            fwdDic  = obj.Dictionary{obj.FORWARD};
            adjDic  = obj.Dictionary{obj.ADJOINT};
            % Measurement process
            msrProc = obj.MeasureProcess;
            
            % Calculation of step size parameter
            framebound = fwdDic.FrameBound;
            step(msrProc,vObs);
            obj.Gamma = 1/(framebound*msrProc.LambdaMax);               

            % Adjoint of measuremnt process
            adjProc = msrProc.clone();
            adjProc.release();
            adjProc.ProcessingMode = 'Adjoint';            
            obj.AdjointProcess = adjProc;
            
            % Initialization
            [obj.X,obj.Scales] = adjDic.step(zeros(size(vObs),'like',vObs));
            obj.Result = fwdDic.step(obj.X,obj.Scales);
        end
        
        function varargout = stepImpl(obj)
            stepImpl@saivdr.restoration.AbstIterativeMethodSystem(obj)
            % Observation
            vObs = obj.Observation;
            % Dictionaries
            fwdDic  = obj.Dictionary{obj.FORWARD};
            adjDic  = obj.Dictionary{obj.ADJOINT};            
            % Measurement process
            msrProc = obj.MeasureProcess;
            adjProc = obj.AdjointProcess;
            %
            scales = obj.Scales;
            gamma  = obj.Gamma;
            lambda = obj.Lambda;
            threshold = gamma*lambda;
            softthresh = @(x) sign(x).*max(abs(x)-threshold,0);
            
            % Previous state
            resPre = obj.Result;
            xPre   = obj.X;
            
            % Main steps 
            t = adjDic.step(adjProc.step(msrProc.step(resPre)-vObs));
            x = softthresh(xPre-gamma*t);
            result = fwdDic(x,scales);
            
            % Output
            if nargout > 0
                varargout{1} = result;
            end
            if nargout > 1
                import saivdr.restoration.AbstIterativeMethodSystem
                varargout{2} = AbstIterativeMethodSystem.rmse(result,resPre);
            end
            
            % Update
            obj.X = x;
            obj.Result = result;
        end        
        
    end
           
    %{
        function setupImpl(obj)
            if obj.IsSizeCompensation
                sizeM = numel(vObs); % 観測データサイズ
                src   = msrProc.step(vObs,'Adjoint');
                coefs = adjDic.step(src); % 変換係数サイズ
                sizeL = numel(coefs);
                obj.LambdaCompensated = obj.Lambda*(sizeM^2/sizeL);
            else
                obj.LambdaCompensated = obj.Lambda;
            end
            lambda_ = obj.LambdaCompensated;
            gamma = 1/lpst; % fの勾配のリプシッツ乗数の逆数
            gdn = PlgGdnSfth('Sigma',sqrt(gamma*lambda_));
            
            % 初期化
            obj.Result = zeros(1,'like',obj.Observation);
            grd0  = zeros(size(vObs)); % 反射率分布の勾配
            if isempty(obj.SplitFactor) % Normal process
                obj.Gamma = gamma;
                obj.GaussianDenoiser = gdn;
                obj.parProc = [];
                %
                fwdDic.release();
                obj.Dictionary{1} = fwdDic.clone();
                adjDic.release();
                obj.Dictionary{2} = adjDic.clone();
                %
                [obj.xpre,obj.scls] = adjDic(grd0); % 変換係数の初期値
            else % OLS/OLA process
                cm = CoefsManipulator(...
                    'Manipulation',...
                    @(t,cpre)  gdn.step(cpre-gamma*t));
                obj.parProc = OlsOlaProcess3d(...
                    'Synthesizer',fwdDic,...
                    'Analyzer',adjDic,...
                    'CoefsManipulator',cm,...
                    'SplitFactor',obj.SplitFactor,...
                    'PadSize',obj.PadSize,...
                    'UseParallel',obj.UseParallel,...
                    'UseGpu',obj.UseGpu,...
                    'IsIntegrityTest',obj.IsIntegrityTest,...
                    'Debug',obj.Debug);
                obj.xpre = obj.parProc.analyze(grd0); % 変換係数の初期値
                obj.parProc.InitialState = obj.xpre;
            end
        end
    
        function varargout = stepImpl(obj)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            vObs = obj.Observation;
            grdFcn_ = obj.grdFcn;
            %
            rpre = obj.Result;
            grd_ = grdFcn_.step(rpre,vObs); % 反射率分布の勾配
            if isempty(obj.SplitFactor) % Normal process
                fwdDic  = obj.Dictionary{1};
                adjDic  = obj.Dictionary{2};
                gdnFcn  = obj.GaussianDenoiser;
                %
                scls_ = obj.scls;
                xpre_ = obj.xpre;
                gamma_ = obj.Gamma;
                %
                t = adjDic.step(grd_); % 分析処理
                x = gdnFcn.step(xpre_-gamma_*t);% 係数操作
                r = fwdDic.step(x,scls_); % 合成処理
                %
                obj.xpre    = x;
            else % OLS/OLA 分析合成処理
                r = obj.parProc(grd_);
            end
            
            % 出力
            if nargout > 0
                varargout{1} = r;
            end
            if nargout > 1
                rmse = norm(r(:)-rpre(:),2)/norm(r(:),2);
                varargout{2} = rmse;
            end
            
            % 状態更新
            obj.Result  = r; % 復元画像
        end
        
    %}
end