classdef PdsHsHcOct3 < matlab.System
    % PDSHSHCOCT3 階層的スパース性とハード制約を利用した主双対近接分離法
    %
    % Output
    %
    %    屈折率率分布
    %
    % Reference
    %
    % - 村松正吾・長山知司・崔　森悦（新潟大）・小野峻佑（東工大）・
    %   太田　岳・任　書晃・日比野　浩（新潟大）
    %   階層的スパース正則化とハード制約を利用したOCTボリュームデータ復元の検討，
    %   電子情報通信学会信号処理研究会，岐阜大，2018年5月
    %
    % - 藤井元暉・村松正吾・崔　森悦（新潟大）・小野峻佑（東工大）・
    %   太田　岳・任　書晃・日比野　浩（新潟大），
    %   階層的スパース正則化とハード制約を利用したOCTボリュームデータ復元の
    %   実データ検証，電子情報通信学会信号処理研究会，拓大文京キャンパス，2018年8月
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
    
    % Public, tunable properties
    properties (Nontunable)
        Observation
        Lambda  = 0.01     % 正則化パラメータ
        Eta     = 0.01     % 正則化パラメータ
        Gamma1  = 0.01     % ステップサイズ
        Gamma2  = []
        Beta    = 0.0      % 忠実項の勾配のリプシッツ定数
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
        %
    end
    
    properties (GetAccess = public, SetAccess = private)
        Result
        LambdaCompensated
        EtaCompensated
    end
    
    properties(Nontunable, Access = private)
        dltFcn
        grdFcn
        parProc
    end
    
    properties(Nontunable, Logical)
        IsIntegrityTest = true
        IsSizeCompensation = false
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
                'KernelMode','Normal',...
                'UseGpu',obj.UseGpu);
            phi_ = RefractIdx2Reflect(...
                'PhiMode',obj.PhiMode,...
                'VRange',obj.VRange,...
                'UseGpu',obj.UseGpu);
            obj.grdFcn = CostEvaluator(...
                'Observation',obj.Observation,...
                'MeasureProcess',obj.MeasureProcess,...
                'RefIdx2Ref',phi_,...
                'OutputMode','Gradient',...
                'UseGpu',obj.UseGpu);
            %
            if isempty(obj.Gamma2)
                tauSqd     = obj.dltFcn.LambdaMax + 1;
                obj.Gamma2 = 1/(1.05*tauSqd)*(1/obj.Gamma1-obj.Beta/2);
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
            obj.Result = s.Result;
            obj.xpre = s.xpre;
            obj.scls = s.scls;
            obj.y1 = s.y1;
            obj.y2 = s.y2;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
            vObs = obj.Observation;
            msrProc = obj.MeasureProcess;            
            fwdDic  = obj.Dictionary{1};
            adjDic  = obj.Dictionary{2};
            %
            if obj.IsSizeCompensation
                sizeM = numel(vObs); % 観測データサイズ
                src   = msrProc.step(vObs,'Adjoint');
                sizeN = numel(src); % 屈折率分布サイズ
                coefs = adjDic.step(src); % 変換係数サイズ
                sizeL = numel(coefs);
                obj.LambdaCompensated = obj.Lambda*(sizeM^2/sizeL);
                obj.EtaCompensated    = obj.Eta*(sizeM^2/sizeN);
            else
                obj.LambdaCompensated = obj.Lambda;
                obj.EtaCompensated    = obj.Eta;
            end
            %
            lambda_ = obj.LambdaCompensated;
            eta_    = obj.EtaCompensated;
            gamma1_ = obj.Gamma1;
            gamma2_ = obj.Gamma2;
            %
            obj.GaussianDenoiser{1}.release();
            obj.GaussianDenoiser{1}.Sigma = sqrt(gamma1_*lambda_);
            obj.GaussianDenoiser{2}.release();
            obj.GaussianDenoiser{2}.Sigma = sqrt(eta_/gamma2_);
            
            %初期化
            obj.y1 = zeros(1,'like',vObs);
            obj.y2 = zeros(1,'like',vObs);
            obj.Result = zeros(1,'like',vObs);
            res0 = zeros(size(vObs),'like',vObs);
            if isempty(obj.SplitFactor) % Normal process
                obj.parProc = [];
                %
                fwdDic.release();
                obj.Dictionary{1} = fwdDic.clone();
                adjDic.release();
                obj.Dictionary{2} = adjDic.clone();
                %
                [obj.xpre,obj.scls] = adjDic(res0); % 変換係数の初期値
            else
                import saivdr.restoration.*
                gdn = obj.GaussianDenoiser{1};
                cm = CoefsManipulator(...
                    'Manipulation',...
                    @(t,cpre)  gdn.step(cpre-gamma1_*t));
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
                obj.xpre = obj.parProc.analyze(res0); % 変換係数の初期値
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
            gamma2_ = obj.Gamma2;
            gdnFcnH = obj.GaussianDenoiser{2};
            %
            y1_  = obj.y1;
            y2_  = obj.y2;
            rpre = obj.Result;
            prx_ = grdFcn_.step(rpre) + (-dltFcn_.step(y1_)) + y2_;
            if isempty(obj.SplitFactor) % Normal process
                fwdDic  = obj.Dictionary{1};
                adjDic  = obj.Dictionary{2};
                gdnFcnG = obj.GaussianDenoiser{1};
                %
                scls_ = obj.scls;
                xpre_ = obj.xpre;
                gamma1_ = obj.Gamma1;
                %
                t = adjDic.step(prx_); % 分析処理
                x = gdnFcnG.step(xpre_-gamma1_*t); % 係数操作
                v = fwdDic.step(x,scls_); % 合成処理
                %
                obj.xpre = x;
            else % OLS/OLA 分析合成処理
                v = obj.parProc.step(prx_);
            end
            u = 2*v - rpre;
            % lines 6-7
            y1_ = y1_ + gamma2_*dltFcn_.step(u);
            y2_ = y2_ + gamma2_*u;
            % line 8
            y1_ = y1_ - gamma2_*gdnFcnH.step( y1_/gamma2_ );
            % line 9
            pcy2 = y2_/gamma2_;
            pcy2((y2_/gamma2_)<vmin) = vmin;
            pcy2((y2_/gamma2_)>vmax) = vmax;
            y2_ = y2_ - gamma2_*pcy2;
            % line 10
            r = (u+rpre)/2;
            
            % 出力
            if nargout > 0
                varargout{1} = r;
            end
            if nargout > 1
                rmse = norm(r(:)-rpre(:),2)/norm(r(:),2);
                varargout{2} = rmse;
            end
            
            % 状態更新
            obj.y1 = y1_;
            obj.y2 = y2_;
            obj.Result = r;
            obj.Iteration = obj.Iteration + 1;
        end
        
        function resetImpl(obj)
            % Initialize / reset discrete-state properties
            obj.Iteration = 0;
        end
    end
  
end