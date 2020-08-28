classdef PixelLossSystem < ...
        saivdr.degradation.linearprocess.AbstLinearSystem %#codegen
    %PIXELLOSSSYSTEM Pixel-loss process
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
        LossType    = 'Random';
        %
        Density = 0.5;
        Seed    = 0;
        Mask = [];
    end
    
    properties (Hidden, Transient)
        LossTypeSet = ...
            matlab.system.StringSet({...
            'Random',...
            'Specified'})
    end
    
    properties (Access = protected, Nontunable)
       maskArray 
    end
    
    methods
        % Constractor
        function obj = PixelLossSystem(varargin)
            obj = ...
                obj@saivdr.degradation.linearprocess.AbstLinearSystem(...
                varargin{:});
        end

    end
    
    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@...
                saivdr.degradation.linearprocess.AbstLinearSystem(obj);
            s.maskArray = obj.maskArray;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.maskArray = s.maskArray;
            loadObjectImpl@...
                saivdr.degradation.linearprocess.AbstLinearSystem(obj,s,wasLocked);
        end
        
        function flag = isInactiveSubPropertyImpl(obj,propertyName)
            if strcmp(propertyName,'Density') || ...
                    strcmp(propertyName,'Seed')
                flag = ~strcmp(obj.LossType,'Random');            
            elseif strcmp(propertyName,'Mask')
                flag = ~strcmp(obj.LossType,'Specified');
            else
                flag = false;
            end
        end
        
        function setupImpl(obj,input)
            
            nDim = getOriginalDimension(obj,size(input));
            if strcmp(obj.DataType,'Image')
                switch obj.LossType
                    case {'Random'}
                        broadcast_id =1;
                        if labindex == broadcast_id
                            rng(obj.Seed,'twister')
                            maskArray_ = rand(nDim(1:2)) > obj.Density;
                            obj.maskArray = labBroadcast(broadcast_id,...
                                maskArray_);
                        else
                            obj.maskArray = labBroadcast(broadcast_id);
                        end
                    case {'Specified'}
                        obj.maskArray = obj.Mask;
                        obj.Density = sum(obj.maskArray(:))/numel(obj.maskArray);
                    otherwise
                        me = MException('SaivDr:InvalidOption',...
                            'Invalid loss type');
                        throw(me);
                end
            else % Volumetric Data
                switch obj.LossType
                    case {'Random'}
                        broadcast_id =1;
                        if labindex == broadcast_id
                            rng(obj.Seed,'twister')
                            maskArray_ = rand(nDim(1:3)) > obj.Density;
                            obj.maskArray = labBroadcast(broadcast_id,...
                                maskArray_);
                        else
                            obj.maskArray = labBroadcast(broadcast_id);
                        end
                    case {'Specified'}
                        obj.maskArray = obj.Mask;
                        obj.Density = sum(obj.maskArray(:))/numel(obj.maskArray);
                    otherwise
                        me = MException('SaivDr:InvalidOption',...
                            'Invalid loss type');
                        throw(me);
                end
            end
            %
            setupImpl@saivdr.degradation.linearprocess. ...
                AbstLinearSystem(obj,input);            
        end
        
        function picture = normalStepImpl(obj,picture)
            if strcmp(obj.DataType,'Image')
                nCmps = size(picture,3);
                for iCmp = 1:nCmps
                    cmp = picture(:,:,iCmp);
                    cmp(obj.maskArray==0)=0;
                    picture(:,:,iCmp) = cmp;
                end
            else % Volumetric Data
                picture(obj.maskArray==0)=0;
            end
        end
        
        function output = adjointStepImpl(obj,input)
            output = normalStepImpl(obj,input);
        end
        
        function originalDim = getOriginalDimension(~,ovservedDim)
            originalDim = ovservedDim;
        end      
        
    end
    
end
