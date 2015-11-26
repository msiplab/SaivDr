classdef UdHaarSynthesis3dSystem < saivdr.dictionary.AbstSynthesisSystem %#codegen
    %UdHaarSynthesis3dSystem Synthesis system for undecimated Haar transform
    %
    % SVN identifier:
    % $Id: UdHaarSynthesis3dSystem.m 683 2015-05-29 08:22:13Z sho $
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
    
    properties (Access = private)
        kernels
        nPixels
        nLevels 
        dim
    end
        
    methods
        function obj = UdHaarSynthesis3dSystem(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
            % 
            obj.kernels.AA(:,:,1) = [ 1 1 ; 1 1 ];   % AA
            obj.kernels.AA(:,:,2) = [ 1 1 ; 1 1 ];   
            obj.kernels.HA(:,:,1) = [ 1 -1 ; 1 -1 ]; % HA
            obj.kernels.HA(:,:,2) = [ 1 -1 ; 1 -1 ]; 
            obj.kernels.VA(:,:,1) = [ 1 1 ; -1 -1 ]; % VA
            obj.kernels.VA(:,:,2) = [ 1 1 ; -1 -1 ]; 
            obj.kernels.DA(:,:,1) = [ 1 -1 ; -1 1 ]; % DA
            obj.kernels.DA(:,:,2) = [ 1 -1 ; -1 1 ]; 
            %
            obj.kernels.AD(:,:,1) = [ 1 1 ; 1 1 ];   % AD
            obj.kernels.AD(:,:,2) = -[ 1 1 ; 1 1 ];   
            obj.kernels.HD(:,:,1) = [ 1 -1 ; 1 -1 ]; % HD
            obj.kernels.HD(:,:,2) = -[ 1 -1 ; 1 -1 ]; 
            obj.kernels.VD(:,:,1) = [ 1 1 ; -1 -1 ]; % VD
            obj.kernels.VD(:,:,2) = -[ 1 1 ; -1 -1 ]; 
            obj.kernels.DD(:,:,1) = [ 1 -1 ; -1 1 ]; % DD
            obj.kernels.DD(:,:,2) = -[ 1 -1 ; -1 1 ]; 
            %
            obj.FrameBound = 1;
        end
    end
    
    methods (Access=protected)
        
        function validateInputsImpl(~, ~, scales)
            % Check scales
            if mod(size(scales,1),7) ~= 1
                error('The number of rows must be 7n+1.')
               
            end
            if nnz(scales-repmat(scales(1,:),[size(scales,1), 1])) > 1
                error('Scales must have the same rows.')
            end
        end
   
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
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
            loadObjectImpl@matlab.System(obj,s,wasLocked); 
        end
               
        function setupImpl(obj, ~, scales)
            obj.dim = scales(1,:);
            obj.nPixels = prod(obj.dim);
            obj.nLevels = (size(scales,1)-1)/7;
        end
        
        function resetImpl(~)
        end
        
        function y = stepImpl(obj, u, ~)
            ufactor = 2^(obj.nLevels-1);
            kernelSize = 2^obj.nLevels;
            weight = 1/(kernelSize^3);
            haa = upsample3_(obj,obj.kernels.AA,ufactor);
            hha = upsample3_(obj,obj.kernels.HA,ufactor);
            hva = upsample3_(obj,obj.kernels.VA,ufactor);
            hda = upsample3_(obj,obj.kernels.DA,ufactor);
            had = upsample3_(obj,obj.kernels.AD,ufactor);
            hhd = upsample3_(obj,obj.kernels.HD,ufactor);
            hvd = upsample3_(obj,obj.kernels.VD,ufactor);
            hdd = upsample3_(obj,obj.kernels.DD,ufactor);
            uaa = reshape(u(1:obj.nPixels),obj.dim);
            uha = reshape(u(obj.nPixels+1:2*obj.nPixels),obj.dim);
            uva = reshape(u(2*obj.nPixels+1:3*obj.nPixels),obj.dim);
            uda = reshape(u(3*obj.nPixels+1:4*obj.nPixels),obj.dim);
            uad = reshape(u(4*obj.nPixels+1:5*obj.nPixels),obj.dim);
            uhd = reshape(u(5*obj.nPixels+1:6*obj.nPixels),obj.dim);
            uvd = reshape(u(6*obj.nPixels+1:7*obj.nPixels),obj.dim);
            udd = reshape(u(7*obj.nPixels+1:8*obj.nPixels),obj.dim);            
            yaa = imfilter(uaa,haa,'conv','circular')*weight;
            yha = imfilter(uha,hha,'conv','circular')*weight;
            yva = imfilter(uva,hva,'conv','circular')*weight;
            yda = imfilter(uda,hda,'conv','circular')*weight;
            yad = imfilter(uad,had,'conv','circular')*weight;
            yhd = imfilter(uhd,hhd,'conv','circular')*weight;
            yvd = imfilter(uvd,hvd,'conv','circular')*weight;
            ydd = imfilter(udd,hdd,'conv','circular')*weight;            
            iSubband = 8;
            for iLevel = 1:obj.nLevels
                if obj.nLevels-iLevel < 2
                    offset = [1 1 1]; 
                else
                    offset = [1 1 1]*2^(obj.nLevels-iLevel-1);
                end
                y = circshift(yaa + yha + yva + yda + yad + yhd + yvd + ydd, offset);                
                ufactor = ufactor/2;
                kernelSize = kernelSize/2;
                weight = 1/(kernelSize^3);
                if iLevel < obj.nLevels
                    haa = upsample3_(obj,obj.kernels.AA,ufactor);
                    hha = upsample3_(obj,obj.kernels.HA,ufactor);
                    hva = upsample3_(obj,obj.kernels.VA,ufactor);
                    hda = upsample3_(obj,obj.kernels.DA,ufactor);
                    had = upsample3_(obj,obj.kernels.AD,ufactor);
                    hhd = upsample3_(obj,obj.kernels.HD,ufactor);
                    hvd = upsample3_(obj,obj.kernels.VD,ufactor);
                    hdd = upsample3_(obj,obj.kernels.DD,ufactor);
                    uaa = y;
                    uha = reshape(u(iSubband*obj.nPixels+1:(iSubband+1)*obj.nPixels),obj.dim);
                    uva = reshape(u((iSubband+1)*obj.nPixels+1:(iSubband+2)*obj.nPixels),obj.dim);
                    uda = reshape(u((iSubband+2)*obj.nPixels+1:(iSubband+3)*obj.nPixels),obj.dim);
                    uad = reshape(u((iSubband+3)*obj.nPixels+1:(iSubband+4)*obj.nPixels),obj.dim);
                    uhd = reshape(u((iSubband+4)*obj.nPixels+1:(iSubband+5)*obj.nPixels),obj.dim);
                    uvd = reshape(u((iSubband+5)*obj.nPixels+1:(iSubband+6)*obj.nPixels),obj.dim);
                    udd = reshape(u((iSubband+6)*obj.nPixels+1:(iSubband+7)*obj.nPixels),obj.dim);
                    yaa = imfilter(uaa,haa,'conv','circular');
                    yha = imfilter(uha,hha,'conv','circular')*weight;
                    yva = imfilter(uva,hva,'conv','circular')*weight;
                    yda = imfilter(uda,hda,'conv','circular')*weight;
                    yad = imfilter(uad,had,'conv','circular')*weight;
                    yhd = imfilter(uhd,hhd,'conv','circular')*weight;
                    yvd = imfilter(uvd,hvd,'conv','circular')*weight;
                    ydd = imfilter(udd,hdd,'conv','circular')*weight;                    
                    iSubband = iSubband + 7;
                end
            end
        end

    end
    
    methods (Access = private)
        
        function value = upsample3_(~,x,ufactor)
            value = shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(x,...
                    ufactor),1),...
                    ufactor),1),...
                    ufactor),1);            
        end
    end
end
