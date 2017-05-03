classdef Order2BuildingBlockTypeII < matlab.System
    %ORDER2BUILDINGBLOCKTYPEII Type-II building block with order 2
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

    properties
        NumberOfSymmetricChannels = 2 
        NumberOfAntisymmetricChannels = 3
    end
    
    properties (Access=protected)        
        nChannels
        Is
        Ia
    end
    
    properties(Access=protected,Logical)
        IsPsGreaterThanPa
    end
    
    methods
        function obj = Order2BuildingBlockTypeII(varargin)
            setProperties(obj,nargin,varargin{:});              
        end 
    end
    
    methods (Access = protected)

        function processTunedPropertiesImpl(obj)        
            propChange = ...
                isChangedProperty(obj,'NumberOfSymmetricChannels') ||...
                isChangedProperty(obj,'NumberOfAntisymmetricChannels');
            if propChange
                ps = obj.NumberOfSymmetricChannels;
                pa = obj.NumberOfAntisymmetricChannels;
                %
                obj.nChannels = ps+pa;
                obj.Is = eye(ps);
                obj.Ia = eye(pa);
                if ps > pa
                    obj.IsPsGreaterThanPa = true;
                else
                    obj.IsPsGreaterThanPa = false;
                end
            end
        end
        
        function setupImpl(obj)
            ps = obj.NumberOfSymmetricChannels;
            pa = obj.NumberOfAntisymmetricChannels;
            %
            obj.nChannels = ps+pa;
            obj.Is = eye(ps);
            obj.Ia = eye(pa);
            if ps > pa
                obj.IsPsGreaterThanPa = true;
            else
                obj.IsPsGreaterThanPa = false;
            end
        end
        
        function output = stepImpl(obj,input,mtxW,mtxU,nshift)
            if obj.IsPsGreaterThanPa
                R = blkdiag(obj.Is,mtxU);
                temp   = R*processQo_(obj,input, nshift);
                R = blkdiag(mtxW,obj.Ia);
                output = R*processQe_(obj,temp, nshift);
            else
                R = blkdiag(mtxW,obj.Ia);
                temp   = R*processQo_(obj,input, nshift);
                R = blkdiag(obj.Is,mtxU);
                output = R*processQe_(obj,temp, nshift);                
            end
        end
        
    end
    
    methods (Access = private)
        
        function value = processQo_(obj,x,nZ_)
            ps = obj.NumberOfSymmetricChannels;
            pa = obj.NumberOfAntisymmetricChannels;
            nChMx = max([ps pa]);
            nChMn = min([ps pa]);
            ch = ps+pa;
            x = butterfly_(obj,x,nChMx,nChMn);
            nLen = size(x,2);
            value = zeros([sum(ch) nLen+nZ_]);
            value(1:nChMn,1:nLen) = x(1:nChMn,:);
            value(nChMn+1:end,nZ_+1:end) = x(nChMn+1:end,:);
            value = butterfly_(obj,value,nChMx,nChMn)/2.0;
        end
        
        function value = processQe_(obj,x,nZ_)
            ps = obj.NumberOfSymmetricChannels;
            pa = obj.NumberOfAntisymmetricChannels;
            nChMx = max([ps pa]);
            nChMn = min([ps pa]);
            ch = ps+pa;
            x = butterfly_(obj,x,nChMx,nChMn);
            nLen = size(x,2);
            value = zeros([sum(ch) nLen+nZ_]);
            value(1:nChMx,1:nLen) = x(1:nChMx,:);
            value(nChMx+1:end,nZ_+1:end) = x(nChMx+1:end,:);
            value = butterfly_(obj,value,nChMx,nChMn)/2.0;
        end
        
        function value = butterfly_(~,x,nChMx,nChMn)
            upper = x(1:nChMn,:);
            middle = x(nChMn+1:nChMx,:);
            lower = x(nChMx+1:end,:);
            value = [
                upper+lower ;
                1.414213562373095*middle;
                upper-lower ];
        end
    end
end
