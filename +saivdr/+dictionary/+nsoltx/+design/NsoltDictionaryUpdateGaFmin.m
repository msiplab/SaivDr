classdef NsoltDictionaryUpdateGaFmin < ...
        saivdr.dictionary.nsoltx.design.AbstNsoltDesignerGaFmin %#~codegen
    %NSOLTDICTIONARYUPDATEGAFMIN Update step of NSOLT dictionary learning 
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2020, Shogo MURAMATSU
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
        NumberOfLevels = 1;
        IsFixedCoefs       = true
    end
    
    properties (Logical)
        IsVerbose = false
    end
    
    properties (Access = protected)
        aprxError
    end
    
    methods
        
        function obj = NsoltDictionaryUpdateGaFmin(varargin)
            obj = obj@saivdr.dictionary.nsoltx.design.AbstNsoltDesignerGaFmin(...
                varargin{:});
        end
        
        function [value, grad] = costFcnAng(obj, lppufb, angles)
            clnlppufb = clone(lppufb);
            clnaprxer = clone(obj.aprxError);
            angles = reshape(angles,obj.sizeAngles);
            set(clnlppufb,'Angles',angles);
            [value,grad] = step(clnaprxer,clnlppufb,...
                obj.SparseCoefficients,obj.SetOfScales);
        end
        
        function value = costFcnMus(obj, lppufb, mus)
            clnlppufb = clone(lppufb);
            clnaprxer = clone(obj.aprxError); 
            mus = 2*(reshape(mus,obj.sizeMus))-1;
            set(clnlppufb, 'Mus', mus);
            value = step(clnaprxer,clnlppufb,...
                obj.SparseCoefficients,obj.SetOfScales);
        end
        
        function value = isConstrained(~,~)
            value = false;
        end        
        
        function [c,ce] = nonlconFcn(~,~,~)
            c  = [];
            ce = [];
        end        
        
    end
    
    methods (Access=protected)

         function [ lppufb, fval, exitflag ] = stepImpl(obj,lppufb_,options)
             if obj.IsVerbose
                fprintf('IsFixedCoefs = %d\n', get(obj.aprxError,'IsFixedCoefs'));
             end                
             [ lppufb, fval, exitflag ] = stepImpl@...
                 saivdr.dictionary.nsoltx.design.AbstNsoltDesignerGaFmin(...
                 obj,lppufb_,options);
         end
        
        function setupImpl(obj,lppufb_,options)
            setupImpl@saivdr.dictionary.nsoltx.design.AbstNsoltDesignerGaFmin(obj,lppufb_,options);
            import saivdr.dictionary.nsoltx.design.AprxErrorWithSparseRep
            obj.aprxError = AprxErrorWithSparseRep(...
                'TrainingImages', obj.TrainingImages,...
                'NumberOfLevels',obj.NumberOfLevels,...
                'GradObj',obj.GradObj,...
                'IsFixedCoefs',obj.IsFixedCoefs);
        end
        
        function validatePropertiesImpl(obj)
            if isempty(obj.TrainingImages)
                error('Training images should be provided');
            elseif ~iscell(obj.TrainingImages)
                error('Training images should be provided as cell data');
            end
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsoltx.design.AbstNsoltDesignerGaFmin(obj);
            s.aprxError = matlab.System.saveObject(obj.aprxError);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.aprxError = matlab.System.loadObject(s.aprxError);
            loadObjectImpl@saivdr.dictionary.nsoltx.design.AbstNsoltDesignerGaFmin(obj,s,wasLocked);
        end   
        
    end

end
