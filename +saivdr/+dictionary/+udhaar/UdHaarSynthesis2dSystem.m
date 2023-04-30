classdef UdHaarSynthesis2dSystem < saivdr.dictionary.AbstSynthesisSystem %#codegen
    %UDHAARSYNTHESIS2DSYSTEM Synthesis system for undecimated Haar transform
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2022, Shogo MURAMATSU
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
        filters
    end
        
    methods
        function obj = UdHaarSynthesis2dSystem(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
            % 
            obj.kernels{1} = [ 1 1 ; 1 1 ]; % A
            obj.kernels{2} = [ 1 -1 ; 1 -1 ]; % H
            obj.kernels{3} = [ 1 1 ; -1 -1 ]; % V
            obj.kernels{4} = [ 1 -1 ; -1 1 ]; % D
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
            s.filters = obj.filters;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.kernels = s.kernels;
            obj.nPixels = s.nPixels;
            obj.nLevels = s.nLevels;
            obj.dim = s.dim;
            obj.filters = s.filters;
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj,s,wasLocked);
        end
               
        function setupImpl(obj, coefs, scales)
            obj.dim = scales(1,:);
            obj.nPixels = prod(obj.dim);
            obj.nLevels = (size(scales,1)-1)/3;
            obj.filters = filters_(obj);            

            % NOTE:
            % imfilter of R2017a has a bug for double precision array
            if strcmp(version('-release'),'2017a') && ...
                    isa(coefs,'double')
                warning(['IMFILTER of R2017a with CIRCULAR option has a bug for double precison array.' ...
                    ' Please visit https://jp.mathworks.com/support/bugreports/ and search #BugID: 1554862.' ])
            end
            
        end

        function resetImpl(~)
        end

        function y = stepImpl(obj, coefs, ~)
            F_ = obj.filters;
            nPixels_ = obj.nPixels;
            dim_ = obj.dim;
            nLevels_ = obj.nLevels;
            nSubbands_ = nLevels_*3 + 1;
            offset = [1 1];
            y = 0;
            for iSubband = 1:nSubbands_
                u = reshape(coefs((iSubband-1)*nPixels_+1:iSubband*nPixels_),dim_);
                y = y + circshift(...
                    imfilter(u,F_{iSubband},'conv','circ'),offset);
            end
        end

    end

    methods (Access = private)
        
        function F = filters_(obj)
            nLevels_ = obj.nLevels;
            F = cell(nLevels_*3+1,1);
            K = obj.kernels;
            % iLevel == nLevels
            ufactor = 2^(nLevels_-1);
            kernelSize = 2^nLevels_;
            weight = 1/(kernelSize^2);
            Ku = cellfun(@(x) filterupsample2_(obj,x,ufactor),K,'UniformOutput',false);
            for idx = 1:4
                F{idx} = Ku{idx}*weight;
            end
            % iLevel < nLevels
            for iLevel = nLevels_-1:-1:1
                ufactor = ufactor/2;
                kernelSize = kernelSize/2;
                weight = 1/(kernelSize^2);
                Ku = cellfun(@(x) filterupsample2_(obj,x,ufactor),K,'UniformOutput',false);
                for iSubband = 1:((nLevels_-iLevel)*3)+1
                    F{iSubband} = convn(F{iSubband},Ku{1}); 
                end
                for idx = 2:4
                    iSubband = (nLevels_- iLevel)*3+idx;
                    F{iSubband} = Ku{idx}*weight;
                end
            end
        end

        function value = filterupsample2_(~,x,ufactor)
            value = shiftdim(upsample(...
                shiftdim(upsample(x,...
                ufactor),1),...
                ufactor),1);
            value = value(1:end-ufactor+1,1:end-ufactor+1);
        end
    end
end
