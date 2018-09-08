classdef PdPnPImRestoration3d < matlab.System
    %PDPNPIMRESTORATION3D 3-D Image Restoration with Plug & Play PDS
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

    % Public, tunable properties
    properties (Nontunable)
        Lambda = 0.01; % Regularization parameter
        Gamma1 = 0.01; % Step size
    end
    %{
    properties (Nontunable)
        
        Observation
        Lambda  = 0.01     % 正則化パラメータ
        Eta     = 0.01     % 正則化パラメータ
        Gamma1  = 0.01     % ステップサイズ
        Gamma2  = []
        VRange  = [ 1.00 1.50 ]  % ハード制約
        PhiMode = 'Linear'       % 線形Φ
        IsNoDcShrink = false     % 直流ソフト閾値処理回避
        IsEnvelopeWeight = false % 包絡線重みづけ
        %
        MeasureProcess
        Dictionary
        GaussianDenoiser
        %
        SplitFactor = []
        PadSize     = [ 0 0 0 ]
    end
    
    properties (GetAccess = public, SetAccess = private)
        Result
    end
     
    properties(Nontunable, Access = private)
        dltFcn
        grdFcn
        parProc
    end

    properties(Nontunable, Logical)
        IsIntegrityTest = true
        UseParallel = false        
        UseGpu = false
    end
    
    properties(Nontunable,Logical, Hidden)
        Debug = false
    end
    
    properties(Access = private)
        y1
        y2
        xpre
        scls
        grd
    end
    
    properties (Hidden)
        PhiModeSet = ...
            matlab.system.StringSet(...
            {'Reflection','Linear','Signed-Quadratic','Identity'});        
    end

    properties(DiscreteState)
        Iteration
    end
  
    methods
        function obj = PdsHsHcOct3(varargin)
            setProperties(obj,nargin,varargin{:})
            %
            obj.dltFcn = Sobel3d(...
                'KernelMode','Normal');
            phi_ = RefractIdx2Reflect(...
                'PhiMode',obj.PhiMode,...
                'VRange',obj.VRange);
            obj.grdFcn = CostEvaluator(...
                'Observation',obj.Observation,...
                'MeasureProcess',obj.MeasureProcess,...
                'RefIdx2Ref',phi_,...
                'OutputMode','Gradient');
            %
            if isempty(obj.Gamma2)
                tauSqd     = obj.dltFcn.LambdaMax + 1;
                obj.Gamma2 = 1/(1.05*obj.Gamma1*tauSqd);
            end
        end
    end
    
    methods(Access = protected)
        
           function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.Result = obj.Result;
            s.y1 = obj.y1;
            s.y2 = obj.y2;
            s.xpre = obj.xpre;
            s.scls = obj.scls;
            s.grd = obj.grd;
            s.dltFcn = matlab.System.saveObject(obj.dltFcn);
            s.grdFcn = matlab.System.saveObject(obj.grdFcn);
            s.parProc = matlab.System.saveObject(obj.parProc);
            s.Dictionary{1} = matlab.System.saveObject(obj.Dictionary{1});
            s.Dictionary{2} = matlab.System.saveObject(obj.Dictionary{2});
            if isLocked(obj)
                s.Iteration = obj.Iteration;
            end            
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            if wasLocked
                obj.Iteration = s.Iteration;
            end            
            obj.Dictionary{1} = matlab.System.loadObject(s.Dictionary{1});
            obj.Dictionary{2} = matlab.System.loadObject(s.Dictionary{2});            
            obj.dltFcn = matlab.System.loadObject(s.dltFcn);
            obj.grdFcn = matlab.System.loadObject(s.grdFcn);
            obj.parProc = matlab.System.loadObject(s.parProc);
            obj.Result = obj.Result;
            obj.xpre = s.xpre;
            obj.scls = s.scls;
            obj.grd = s.grd;
            obj.y1 = s.y1;
            obj.y2 = s.y2;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
            %vObs = obj.Observation;
            lambda_ = obj.Lambda;
            eta_    = obj.Eta;
            gamma1_ = obj.Gamma1;
            gamma2_ = obj.Gamma2;            
            %
            fwdDic  = obj.Dictionary{1};            
            adjDic  = obj.Dictionary{2};
            obj.GaussianDenoiser{1}.release();
            obj.GaussianDenoiser{1}.Sigma = sqrt(gamma1_*lambda_);
            obj.GaussianDenoiser{2}.release();
            obj.GaussianDenoiser{2}.Sigma = sqrt(eta_/gamma2_);
            
            
            %初期化
            obj.y1 = zeros(1,'like',obj.Observation);
            obj.y2 = zeros(1,'like',obj.Observation);
            obj.Result = zeros(1,'like',obj.Observation);
            obj.grd = zeros(size(obj.Observation),'like',obj.Observation);
            if isempty(obj.SplitFactor) % Normal process
                obj.parProc = [];
                %
                fwdDic.release();
                obj.Dictionary{1} = fwdDic.clone();
                adjDic.release();
                obj.Dictionary{2} = adjDic.clone();
                %
                [obj.xpre,obj.scls] = adjDic(obj.grd); % 変換係数の初期値
            else
                gdn = obj.GaussianDenoiser{1};
                cm = CoefsManipulator(...
                    'Manipulation',...
                    @(x,s) obj.manipulation(x,s,@(x) gdn.step(x),gamma1_));
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
                obj.xpre = obj.parProc.analyze(obj.grd); % 変換係数の初期値
                obj.parProc.InitialState = obj.xpre;
            end
        end
        
        function varargout = stepImpl(obj)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            dltFcn_ = obj.dltFcn;
            grdFcn_ = obj.grdFcn;
            %
            vmin = obj.VRange(1);
            vmax = obj.VRange(2);
            gamma1_ = obj.Gamma1;
            gamma2_ = obj.Gamma2;
            gdnFcnH = obj.GaussianDenoiser{2};            
            %
            y1_  = obj.y1;
            y2_  = obj.y2;
            grd_ = obj.grd;
            wpre = obj.Result;
            if isempty(obj.SplitFactor) % Normal process
                fwdDic  = obj.Dictionary{1};
                adjDic  = obj.Dictionary{2};
                gdnFcnG = obj.GaussianDenoiser{1};
                %
                scls_ = obj.scls;
                xpre_ = obj.xpre;
                %
                t = adjDic.step(grd_); % 分析処理
                x = gdnFcnG.step(xpre_-gamma1_*t); % 係数操作
                v = 2*x - xpre_;
                u = fwdDic.step(v,scls_); % 合成処理
                %
                obj.xpre = x;
            else % OLS/OLA 分析合成処理
                u = obj.parProc.step(obj.grd);
            end
            % lines 5-6
            y1_ = y1_ + gamma2_*dltFcn_.step(u);
            y2_ = y2_ + gamma2_*u;
            % line 7
            y1_ = y1_ - gamma2_*gdnFcnH.step( y1_/gamma2_ );
            % line 8
            pcy2 = y2_/gamma2_;
            pcy2((y2_/gamma2_)<vmin) = vmin;
            pcy2((y2_/gamma2_)>vmax) = vmax;
            y2_ = y2_ - gamma2_*pcy2;

            % 分析合成システムを利用するため，最終ステップに移動
            % ※grdFcn は辞書をふくまない
            % line 2 (dltFcn の随伴作用素は -dltFcn)
            %w = Dx が欲しい（合成処理）．演算量削減のために変形
            %
            %u = D(2x - xpre) は計算済み
            % ↓
            %u = 2Dx - Dxpre = 2w - wpre
            % ↓
            % wpre は前の繰り返しで Result として保存
            w = (u+wpre)/2;
            grd_ = grdFcn_.step(w) + (-dltFcn_.step(y1_)) + y2_;

            % 出力
            if nargout > 0
                varargout{1} = u;
            end
            if nargout > 1
                rmse = norm(u(:)-obj.Result(:),2)/norm(u(:),2);
                varargout{2} = rmse;
            end
            
            % 状態更新
            obj.y1 = y1_;
            obj.y2 = y2_;
            obj.grd = grd_;
            obj.Result = w;
            obj.Iteration = obj.Iteration + 1;
        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
            obj.Iteration = 0;
        end
    end
    
    methods(Static)
        function [y,s] = manipulation(x,s,gdnFcn,gamma)
            u = s-gamma*x;            
            v = gdnFcn(u);
            y = 2*v-s;
            s = v;
        end
    end
    %}
end