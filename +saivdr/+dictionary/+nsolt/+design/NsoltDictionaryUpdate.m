classdef NsoltDictionaryUpdate < saivdr.dictionary.nsolt.design.AbstNsoltDesigner %#~codegen
    %NSOLTDICTIONARYUPDATE Update step of NSOLT dictionary learning 
    %
    % SVN identifier:
    % $Id: NsoltDictionaryUpdate.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
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
    % http://msiplab.eng.niigata-u.ac.jp/    
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
    
    properties (Access = protected)
        aprxError
    end
    
    methods
        
        function obj = NsoltDictionaryUpdate(varargin)
            import saivdr.dictionary.nsolt.design.AprxErrorWithSparseRep
            obj = obj@saivdr.dictionary.nsolt.design.AbstNsoltDesigner(...
                varargin{:});
            obj.aprxError = AprxErrorWithSparseRep(...
                'SourceImages', obj.SourceImages,...
                'NumberOfTreeLevels',obj.NumberOfTreeLevels,...
                'IsFixedCoefs',obj.IsFixedCoefs);
        end
        
        function value = costFcnAng(obj, lppufb, angles)
            clnlppufb = clone(lppufb);
            clnaprxer = clone(obj.aprxError);
            angles = reshape(angles,obj.sizeAngles);
            set(clnlppufb,'Angles',angles);
            value = step(clnaprxer,clnlppufb,...
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
        
        function validatePropertiesImpl(obj)
            if isempty(obj.SourceImages)
                error('Source images should be provided');
            elseif ~iscell(obj.SourceImages)
                error('Source images should be provided as cell data');
            end
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsolt.design.AbstNsoltDesigner(obj);
            s.aprxError = matlab.System.saveObject(obj.aprxError);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.aprxError = matlab.System.loadObject(s.aprxError);
            loadObjectImpl@saivdr.dictionary.nsolt.design.AbstNsoltDesigner(obj,s,wasLocked);
        end   
        
    end

end
