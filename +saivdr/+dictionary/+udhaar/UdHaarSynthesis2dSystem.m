classdef UdHaarSynthesis2dSystem < saivdr.dictionary.AbstSynthesisSystem %#codegen
    %UDHAARSYNTHESIS2DSYSTEM Synthesis system for undecimated Haar transform
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2018, Shogo MURAMATSU
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
        BoundaryOperation = 'Circular'
    end
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Circular'});
    end
    
    properties (Access = private)
        kernels
        nPixels
        nLevels 
        dim
    end
        
    methods
        function obj = UdHaarSynthesis2dSystem(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
            % 
            obj.kernels.A = [ 1 1 ; 1 1 ]; % A
            obj.kernels.H = [ 1 -1 ; 1 -1 ]; % H
            obj.kernels.V = [ 1 1 ; -1 -1 ]; % V
            obj.kernels.D = [ 1 -1 ; -1 1 ]; % D
            %
            obj.FrameBound = 1;
        end
    end
    
    methods (Access=protected)
        
        function validateInputsImpl(~, ~, scales)
            % Check scales
            if mod(size(scales,1),3) ~= 1
                error('The number of rows must be 3n+1.')
               
            end
            if nnz(scales-repmat(scales(1,:),[size(scales,1), 1])) > 1
                error('Scales must have the same rows.')
            end
        end
   
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj);
            % Save the child System objects            
            s.kernels = obj.kernels;
            s.nPixels = obj.nPixels;
            s.nLevels = obj.nLevels;
            s.dim = obj.dim;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.kernels = s.kernels;
            obj.nPixels = s.nPixels;
            obj.nLevels = s.nLevels;
            obj.dim = s.dim;
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj,s,wasLocked);
        end
               
        function setupImpl(obj, ~, scales)
            obj.dim = scales(1,:);
            obj.nPixels = prod(obj.dim);
            obj.nLevels = (size(scales,1)-1)/3;
        end
        
        function resetImpl(~)
        end
        
        function y = stepImpl(obj, u, ~)
            % NOTE:
            % imfilter of R2017a has a bug for double precision array            
            if strcmp(version('-release'),'2017a') && ...
                    isa(u,'double')
                warning(['IMFILTER of R2017a with CIRCULAR option has a bug for double precison array.' ...
                    ' Please visit https://jp.mathworks.com/support/bugreports/ and search #BugID: 1554862.' ])
            end           
            ufactor = 2^(obj.nLevels-1);
            kernelSize = 2^obj.nLevels;
            weight = 1/(kernelSize^2);
            ha = upsample(upsample(obj.kernels.A,ufactor).',ufactor).';
            hh = upsample(upsample(obj.kernels.H,ufactor).',ufactor).';
            hv = upsample(upsample(obj.kernels.V,ufactor).',ufactor).';
            hd = upsample(upsample(obj.kernels.D,ufactor).',ufactor).';
            ua = reshape(u(1:obj.nPixels),obj.dim);
            uh = reshape(u(obj.nPixels+1:2*obj.nPixels),obj.dim);
            uv = reshape(u(2*obj.nPixels+1:3*obj.nPixels),obj.dim);
            ud = reshape(u(3*obj.nPixels+1:4*obj.nPixels),obj.dim);
            ya = imfilter(ua,ha,'conv','circular')*weight;
            yh = imfilter(uh,hh,'conv','circular')*weight;
            yv = imfilter(uv,hv,'conv','circular')*weight;
            yd = imfilter(ud,hd,'conv','circular')*weight;
            iSubband = 4;
            for iLevel = 1:obj.nLevels
                if obj.nLevels-iLevel < 2
                    offset = [1 1]; 
                else
                    offset = [1 1]*2^(obj.nLevels-iLevel-1);
                end
                y = circshift(ya + yh + yv + yd,offset);                
                ufactor = ufactor/2;
                kernelSize = kernelSize/2;
                weight = 1/(kernelSize^2);                
                if iLevel < obj.nLevels
                    ha = upsample(upsample(obj.kernels.A,ufactor).',ufactor).';
                    hh = upsample(upsample(obj.kernels.H,ufactor).',ufactor).';
                    hv = upsample(upsample(obj.kernels.V,ufactor).',ufactor).';
                    hd = upsample(upsample(obj.kernels.D,ufactor).',ufactor).';
                    ua = y;
                    uh = reshape(u(iSubband*obj.nPixels+1:(iSubband+1)*obj.nPixels),obj.dim);
                    uv = reshape(u((iSubband+1)*obj.nPixels+1:(iSubband+2)*obj.nPixels),obj.dim);
                    ud = reshape(u((iSubband+2)*obj.nPixels+1:(iSubband+3)*obj.nPixels),obj.dim);
                    ya = imfilter(ua,ha,'conv','circular');
                    yh = imfilter(uh,hh,'conv','circular')*weight;
                    yv = imfilter(uv,hv,'conv','circular')*weight;
                    yd = imfilter(ud,hd,'conv','circular')*weight;
                    iSubband = iSubband + 3;
                end
            end
        end

    end
end
