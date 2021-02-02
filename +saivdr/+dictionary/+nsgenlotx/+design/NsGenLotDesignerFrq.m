classdef NsGenLotDesignerFrq < ...
        saivdr.dictionary.nsoltx.design.AbstNsoltDesignerGaFmin %#~codegen
    %NSGENLOTDESIGNERFRQ NS-GenLOT design class with frequency specifications
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2017, Shogo MURAMATSU
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
 
    properties (Nontunable)
        AmplitudeSpecs
        CostClass = 'PassBandErrorStopBandEnergy'
    end
    
    properties (Hidden, Transient)
        CostClassSet = ...
            matlab.system.StringSet({...
            'PassBandErrorStopBandEnergy',...
            'AmplitudeErrorEnergy'})
    end
    
    properties (Access = protected)
        costObj
    end
    
    methods
        function obj = NsGenLotDesignerFrq(varargin)
            obj = obj@saivdr.dictionary.nsoltx.design.AbstNsoltDesignerGaFmin(...
                varargin{:});
            if strcmp(obj.CostClass,'PassBandErrorStopBandEnergy')
                import saivdr.dictionary.nsoltx.design.PassBandErrorStopBandEnergy
                obj.costObj = PassBandErrorStopBandEnergy(...
                    'AmplitudeSpecs',obj.AmplitudeSpecs,...
                    'EvaluationMode','All');
            elseif strcmp(obj.CostClass,'AmplitudeErrorEnergy')
                import saivdr.dictionary.nsoltx.design.AmplitudeErrorEnergy
                obj.costObj = AmplitudeErrorEnergy(...
                    'AmplitudeSpecs',obj.AmplitudeSpecs,...
                    'EvaluationMode','All');
            else
                error('SaivDr:Invalid cost class')
            end
        end
        
        function value = costFcnAng(obj, lppufb, angles)
            clnlppufb = clone(lppufb);
            clnpbesbe = clone(obj.costObj);
            angles = reshape(angles,obj.sizeAngles);
            set(clnlppufb,'Angles',angles);
            value = step(clnpbesbe,clnlppufb);
        end
        
        function value = costFcnMus(obj, lppufb, mus)
            clnlppufb = clone(lppufb);
            clnpbesbe = clone(obj.costObj);
            mus = 2*(reshape(mus,obj.sizeMus))-1;
            set(clnlppufb,'Mus',mus);
            value = step(clnpbesbe,clnlppufb);
        end
        
        function value = isConstrained(~, lppufb)
            if isa(lppufb,'saivdr.dictionary.nsgenlotx.LpPuFb2dVm2System') || ...
                    isa(lppufb,'saivdr.dictionary.nsgenlotx.LpPuFb2dTvmSystem') 
                value = true;
            else
                value = false;
            end
        end
        
        function [c, ce] = nonlconFcn(obj,lppufb,angles)
            clnlppufb = clone(lppufb);
            angles = reshape(angles,obj.sizeAngles);
            step(clnlppufb,angles,[]);
            if isa(clnlppufb,...
                    'saivdr.dictionary.nsgenlotx.LpPuFb2dVm2System') 
                c(1) = get(clnlppufb,'lenx3y')-2;
                c(2) = get(clnlppufb,'lenx3x')-2;
                c(3) = get(clnlppufb,'lambdaxueq');
                c(4) = get(clnlppufb,'lambdayueq');
                ce   = [];
            elseif isa(clnlppufb,...
                    'saivdr.dictionary.nsgenlotx.LpPuFb2dTvmSystem')
                c(1) = get(clnlppufb,'lenx3')-2;
                c(2) = get(clnlppufb,'lambdaueq');
                ce   = [];
            else
                c  = [];
                ce = [];
            end
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsoltx.design.AbstNsoltDesignerGaFmin(obj);
            s.costObj = matlab.System.saveObject(obj.costObj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.costObj = matlab.System.loadObject(s.costObj);
            loadObjectImpl@saivdr.dictionary.nsoltx.design.AbstNsoltDesignerGaFmin(obj,s,wasLocked);
        end
        
        function setupImpl(obj,lppufb,~)
            setupImpl@saivdr.dictionary.nsoltx.design.AbstNsoltDesignerGaFmin(obj,lppufb,[])
            if isConstrained(obj,lppufb) && ...
                    strcmp(char(obj.OptimizationFunction),'fminunc')
                obj.OptimizationFunction = @fmincon;
            end
        end        
        
    end
end
