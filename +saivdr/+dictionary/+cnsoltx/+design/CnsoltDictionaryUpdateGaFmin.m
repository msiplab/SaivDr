classdef CnsoltDictionaryUpdateGaFmin < ...
        saivdr.dictionary.cnsoltx.design.AbstCnsoltDesignerGaFmin %#~codegen
    %NSOLTDICTIONARYUPDATEGAFMIN Update step of NSOLT dictionary learning 
    %
    % SVN identifier:
    % $Id: NsoltDictionaryUpdateGaFmin.m 866 2015-11-24 04:29:42Z sho $
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2014-2015, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627    
    %
    
    properties (Nontunable)
        NumberOfTreeLevels = 1;
        IsFixedCoefs       = true
        SourceImages
    end
    
    properties
        SparseCoefficients
        SetOfScales
    end
    
    properties (Logical)
        IsVerbose = false
    end
    
    properties (Access = protected)
        aprxError
    end
    
    methods
        
        function obj = CnsoltDictionaryUpdateGaFmin(varargin)
            obj = obj@saivdr.dictionary.cnsoltx.design.AbstCnsoltDesignerGaFmin(...
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
                 saivdr.dictionary.cnsoltx.design.AbstCnsoltDesignerGaFmin(...
                 obj,lppufb_,options);
         end
        
        function setupImpl(obj,lppufb_,options)
            setupImpl@saivdr.dictionary.cnsoltx.design.AbstCnsoltDesignerGaFmin(obj,lppufb_,options);
            import saivdr.dictionary.cnsoltx.design.AprxErrorWithSparseRep
            obj.aprxError = AprxErrorWithSparseRep(...
                'SourceImages', obj.SourceImages,...
                'NumberOfTreeLevels',obj.NumberOfTreeLevels,...
                'GradObj',obj.GradObj,...
                'IsFixedCoefs',obj.IsFixedCoefs);
        end
        
        function validatePropertiesImpl(obj)
            if isempty(obj.SourceImages)
                error('Source images should be provided');
            elseif ~iscell(obj.SourceImages)
                error('Source images should be provided as cell data');
            end
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.cnsoltx.design.AbstCnsoltDesignerGaFmin(obj);
            s.aprxError = matlab.System.saveObject(obj.aprxError);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.aprxError = matlab.System.loadObject(s.aprxError);
            loadObjectImpl@saivdr.dictionary.cnsoltx.design.AbstCnsoltDesignerGaFmin(obj,s,wasLocked);
        end   
        
    end

end
