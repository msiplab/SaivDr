classdef UdHaarAnalysis3dSystem < saivdr.dictionary.AbstAnalysisSystem %#codegen
    %UDHAARANALYZSISSYSTEM Analysis system for undecimated Haar transform
    %
    % SVN identifier:
    % $Id: UdHaarAnalysis3dSystem.m 683 2015-05-29 08:22:13Z sho $
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
        coefs
        nPixels
    end

    methods
        function obj = UdHaarAnalysis3dSystem(varargin)
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
        end
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.kernels = obj.kernels;
            s.coefs = obj.coefs;
            s.nPixels = obj.nPixels;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.kernels = s.kernels;
            obj.coefs = s.coefs;
            obj.nPixels = s.nPixels;
            loadObjectImpl@matlab.System(obj,s,wasLocked); 
        end
        
        function setupImpl(obj,u,nLevels)
            obj.nPixels = numel(u);
            obj.coefs = zeros(1,(7*nLevels+1)*obj.nPixels);
        end
        
        function resetImpl(~)
        end
        
        function [ coefs, scales ] = stepImpl(obj, u, nLevels)
            scales = repmat(size(u),[7*nLevels+1, 1]);
            % NOTE:
            % imfilter of R2017a has a bug for double precision array            
            if strcmp(version('-release'),'2017a') && ...
                    isa(u,'double')
                 warning('IMFILTER of R2017a with CIRCULAR option has a bug for double precison array.')
            end               
            yaa = u;          
            hdd = obj.kernels.DD;
            hvd = obj.kernels.VD;
            hhd = obj.kernels.HD;
            had = obj.kernels.AD;
            hda = obj.kernels.DA;
            hva = obj.kernels.VA;            
            hha = obj.kernels.HA;
            haa = obj.kernels.AA;
            iSubband = 7*nLevels+1;
            for iLevel = 1:nLevels
                kernelSize = 2^iLevel;
                weight = 1/(kernelSize^3);
                if iLevel < 2 
                    offset = [0 0 0]; % 1
                else
                    offset = -[1 1 1]*(2^(iLevel-2)-1);
                end
                ydd = circshift(imfilter(yaa,hdd,'corr','circular'),offset);
                yvd = circshift(imfilter(yaa,hvd,'corr','circular'),offset);
                yhd = circshift(imfilter(yaa,hhd,'corr','circular'),offset);                
                yad = circshift(imfilter(yaa,had,'corr','circular'),offset);
                yda = circshift(imfilter(yaa,hda,'corr','circular'),offset);
                yva = circshift(imfilter(yaa,hva,'corr','circular'),offset);
                yha = circshift(imfilter(yaa,hha,'corr','circular'),offset);                
                yaa = circshift(imfilter(yaa,haa,'corr','circular'),offset);
                obj.coefs((iSubband-1)*obj.nPixels+1:iSubband*obj.nPixels) = ...
                    ydd(:).'*weight;
                obj.coefs((iSubband-2)*obj.nPixels+1:(iSubband-1)*obj.nPixels) = ...
                    yvd(:).'*weight;
                obj.coefs((iSubband-3)*obj.nPixels+1:(iSubband-2)*obj.nPixels) = ...
                    yhd(:).'*weight;
                obj.coefs((iSubband-4)*obj.nPixels+1:(iSubband-3)*obj.nPixels) = ...
                    yad(:).'*weight;
                obj.coefs((iSubband-5)*obj.nPixels+1:(iSubband-4)*obj.nPixels) = ...
                    yda(:).'*weight;
                obj.coefs((iSubband-6)*obj.nPixels+1:(iSubband-5)*obj.nPixels) = ...
                    yva(:).'*weight;
                obj.coefs((iSubband-7)*obj.nPixels+1:(iSubband-6)*obj.nPixels) = ...
                    yha(:).'*weight;
                iSubband = iSubband - 7;
                hdd = upsample3_(obj,hdd);
                hvd = upsample3_(obj,hvd);
                hhd = upsample3_(obj,hhd);
                had = upsample3_(obj,had);
                hda = upsample3_(obj,hda);
                hva = upsample3_(obj,hva);
                hha = upsample3_(obj,hha);
                haa = upsample3_(obj,haa);
            end
            obj.coefs(1:obj.nPixels) = yaa(:).'*weight;
            coefs = obj.coefs;
        end
    end
    
    methods (Access = private)
        
        function value = upsample3_(~,x)
            ufactor = 2;
            value = shiftdim(upsample(...
                    shiftdim(upsample(...
                    shiftdim(upsample(x,...
                    ufactor),1),...
                    ufactor),1),...
                    ufactor),1);
        end
    end
end
