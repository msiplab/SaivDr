classdef UdHaarAnalysis2dSystem < saivdr.dictionary.AbstAnalysisSystem %#codegen
    %UDHAARANALYZSISSYSTEM Analysis system for undecimated Haar transform
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
        BoundaryOperation = 'Circular'
    end
    
    properties (Nontunable, PositiveInteger)
        NumberOfLevels = 1        
    end        
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Circular'});
    end
    
    properties (Access = private)
        kernels
        coefs
        nPixels
    end

    methods
        function obj = UdHaarAnalysis2dSystem(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
            % 
            obj.kernels.A = [ 1 1 ; 1 1 ]; % A
            obj.kernels.H = [ 1 -1 ; 1 -1 ]; % H
            obj.kernels.V = [ 1 1 ; -1 -1 ]; % V
            obj.kernels.D = [ 1 -1 ; -1 1 ]; % D
        end
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj);
            s.kernels = obj.kernels;
            s.coefs = obj.coefs;
            s.nPixels = obj.nPixels;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.kernels = s.kernels;
            obj.coefs = s.coefs;
            obj.nPixels = s.nPixels;
            loadObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj,s,wasLocked); 
        end
        
        function setupImpl(obj,u)
            nLevels = obj.NumberOfLevels;
            obj.nPixels = numel(u);
            obj.coefs = zeros(1,(3*nLevels+1)*obj.nPixels,'like',u);
        end
        
        function resetImpl(~)
        end
        
        function [ coefs, scales ] = stepImpl(obj, u)
            nPixels_ = obj.nPixels;
            nLevels = obj.NumberOfLevels;
            scales = repmat(size(u),[3*nLevels+1, 1]);
            % NOTE:
            % imfilter of R2017a has a bug for double precision array
            if strcmp(version('-release'),'2017a') && ...
                    isa(u,'double')
                warning(['IMFILTER of R2017a with CIRCULAR option has a bug for double precison array.' ...
                    ' Please visit https://jp.mathworks.com/support/bugreports/ and search #BugID: 1554862.' ])
            end
            ya = u;
            hd = obj.kernels.D;
            hv = obj.kernels.V;
            hh = obj.kernels.H;
            ha = obj.kernels.A;
            iSubband = 3*nLevels+1;
            for iLevel = 1:nLevels
                kernelSize = 2^iLevel;
                weight = 1/(kernelSize^2);
                if iLevel < 2
                    offset = [0 0]; % 1
                else
                    offset = -[1 1]*(2^(iLevel-2)-1);
                end
                yd = circshift(imfilter(ya,hd,'corr','circular'),offset);
                yv = circshift(imfilter(ya,hv,'corr','circular'),offset);
                yh = circshift(imfilter(ya,hh,'corr','circular'),offset);
                ya = circshift(imfilter(ya,ha,'corr','circular'),offset);
                obj.coefs((iSubband-1)*nPixels_+1:iSubband*nPixels_) = ...
                    yd(:).'*weight;
                obj.coefs((iSubband-2)*nPixels_+1:(iSubband-1)*nPixels_) = ...
                    yv(:).'*weight;
                obj.coefs((iSubband-3)*nPixels_+1:(iSubband-2)*nPixels_) = ...
                    yh(:).'*weight;
                iSubband = iSubband - 3;
                hd = upsample2_(obj,hd);
                hv = upsample2_(obj,hv);
                hh = upsample2_(obj,hh);
                ha = upsample2_(obj,ha);
            end
            obj.coefs(1:nPixels_) = ya(:).'*weight;
            coefs = obj.coefs;
        end

    end
    
    methods (Access = private)
        
        function value = upsample2_(~,x)
            ufactor = 2;
            value = shiftdim(upsample(...
                shiftdim(upsample(x,...
                ufactor),1),...
                ufactor),1);
        end
    end
    
end