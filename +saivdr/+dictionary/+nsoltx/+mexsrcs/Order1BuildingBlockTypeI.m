classdef Order1BuildingBlockTypeI < matlab.System  %#codegen
    %ORDER1BUILDINGBLOCKTYPEI  Type-I building block with order 1
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
            HalfNumberOfChannels
    end
    
    properties (Access=protected)
        nChannels
        I
    end
    
    methods
        function obj = Order1BuildingBlockTypeI(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access = protected)
        
        function processTunedPropertiesImpl(obj)
            propChange = isChangedProperty(obj,'HalfNumberOfChannels');
            if propChange
                p = obj.HalfNumberOfChannels;
                %
                obj.nChannels = 2*p;
                obj.I = eye(p);
            end
        end
        
        
        function setupImpl(obj)
            p = obj.HalfNumberOfChannels;
            %
            obj.nChannels = 2*p;
            obj.I = eye(p);
        end
        
        function output = stepImpl(obj,input,mtxU,nshift)
            R = blkdiag(obj.I,mtxU);
            output = R*processQ_(obj,input,nshift);
        end
    end
    
    methods (Access = private)

        function value = processQ_(obj,x,nZ_)
            hChs = obj.HalfNumberOfChannels;
            nChs = obj.nChannels;
            nLen = size(x,2);
            x = butterfly_(obj,x);
            value = zeros([nChs nLen+nZ_]);
            value(1:hChs,1:nLen) = x(1:hChs,:);
            value(hChs+1:end,nZ_+1:end) = x(hChs+1:end,:);
            value = butterfly_(obj,value)/2.0;
        end
        
        function value = butterfly_(obj,x)
            hChs = obj.HalfNumberOfChannels;
            upper = x(1:hChs,:);
            lower = x(hChs+1:end,:);
            value = [
                upper+lower ;
                upper-lower ];
        end
        
    end
end
