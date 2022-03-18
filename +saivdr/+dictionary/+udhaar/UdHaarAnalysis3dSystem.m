classdef UdHaarAnalysis3dSystem < saivdr.dictionary.AbstAnalysisSystem %#codegen
    %UDHAARANALYSIS3DSYSTEM Analysis system for undecimated Haar transform
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2018-2022, Shogo MURAMATSU
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
    
    properties (Logical)
        UseParallel = false
    end
    
    properties (Nontunable, Access = private)
        kernels
        coefs
        nPixels
        nWorkers
        filters
    end
    
    methods
        
        function obj = UdHaarAnalysis3dSystem(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
            %
            K = cell(8,1);
            ha = double([1  1]);
            hd = double([1 -1]);
            K{1} = { ha, ha, ha };
            K{2} = { ha, hd, ha };
            K{3} = { hd, ha, ha };
            K{4} = { hd, hd, ha };
            K{5} = { ha, ha, hd };
            K{6} = { ha, hd, hd };
            K{7} = { hd, ha, hd };
            K{8} = { hd, hd, hd };                
            %{
            for idx = 1:8
                K{idx} = double(zeros([2 2 2]));
            end
            %
            K{1}(:,:,1) = double([ 1 1 ; 1 1 ]);   % AA
            K{1}(:,:,2) = double([ 1 1 ; 1 1 ]);
            K{2}(:,:,1) = double([ 1 -1 ; 1 -1 ]); % HA
            K{2}(:,:,2) = double([ 1 -1 ; 1 -1 ]);
            K{3}(:,:,1) = double([ 1 1 ; -1 -1 ]); % VA
            K{3}(:,:,2) = double([ 1 1 ; -1 -1 ]);
            K{4}(:,:,1) = double([ 1 -1 ; -1 1 ]); % DA
            K{4}(:,:,2) = double([ 1 -1 ; -1 1 ]);
            %
            K{5}(:,:,1) = double([ 1 1 ; 1 1 ]);   % AD
            K{5}(:,:,2) = double(-[ 1 1 ; 1 1 ]);
            K{6}(:,:,1) = double([ 1 -1 ; 1 -1 ]); % HD
            K{6}(:,:,2) = double(-[ 1 -1 ; 1 -1 ]);
            K{7}(:,:,1) = double([ 1 1 ; -1 -1 ]); % VD
            K{7}(:,:,2) = double(-[ 1 1 ; -1 -1 ]);
            K{8}(:,:,1) = double([ 1 -1 ; -1 1 ]); % DD
            K{8}(:,:,2) = double(-[ 1 -1 ; -1 1 ]);
            %}
            obj.kernels = K;
        end
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj);
            s.kernels = obj.kernels;
            s.coefs = obj.coefs;
            s.nPixels = obj.nPixels;
            s.nWorkers = obj.nWorkers;
            s.filters = obj.filters;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.nWorkers = s.nWorkers;            
            obj.kernels = s.kernels;
            obj.coefs = s.coefs;
            obj.nPixels = s.nPixels;
            obj.filters = s.filters;
            loadObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj,s,wasLocked);
        end
        
        function setupImpl(obj,u)
            nLevels = obj.NumberOfLevels;
            obj.nPixels = numel(u);
            if isa(u,'gpuArray')
                obj.UseGpu = true;
            end
            if obj.UseGpu
                u = gpuArray(u);
            end
            obj.coefs = zeros(1,(7*nLevels+1)*obj.nPixels,'like',u);
            
            if obj.UseParallel
                obj.nWorkers = Inf;
            else
                obj.nWorkers = 0;
            end
            % Spatial domain
            obj.filters = filters_(obj);
            % Frequency domain
            %{
            F_ = filters_(obj);
            nSubbands_ = nLevels*7 + 1;
            dim = size(u);
            Z_ = zeros(dim);
            for iSubband=1:nSubbands_
                Fext = Z_;
                Forg = F_{iSubband};
                szF = size(Forg);
                Fext(1:szF(1),1:szF(2),1:szF(3)) = Forg;
                Fext = circshift(Fext,[1 1 1]-floor(szF/2));
                obj.filters{iSubband} = fftn(Fext);
            end
            %}
            % NOTE:
            % imfilter of R2017a has a bug for double precision array
            if strcmp(version('-release'),'2017a') && ...
                    isa(u,'double')
                warning(['IMFILTER of R2017a with CIRCULAR option has a bug for double precison array.' ...
                    ' Please visit https://jp.mathworks.com/support/bugreports/ and search #BugID: 1554862.' ])
            end
        end
        
        function resetImpl(~)
        end
        
        
        function [ coefs_, scales ] = stepImpl(obj, u)
            if obj.UseGpu
                u = gpuArray(u);
            end
            nLevels_ = obj.NumberOfLevels;
            nPixels_ = obj.nPixels;
            coefs_ = obj.coefs;
            %
            F_ = obj.filters;
            scales = repmat(size(u),[7*nLevels_+1, 1]);   
            nSubbands_ = nLevels_*7 + 1;
            Y = cell(nSubbands_,1);
            %U = fftn(u);
            parfor (iSubband = 1:nSubbands_, obj.nWorkers)
                v = imfilter(u,F_{iSubband}{1}(:),'corr','circ');
                h = imfilter(shiftdim(v,1),F_{iSubband}{2}(:),'corr','circ');
                d = imfilter(shiftdim(h,1),F_{iSubband}{3}(:),'corr','circ');
                Y{iSubband} = shiftdim(d,1);
                % Spatial domain
                %Y{iSubband} = imfilter(u,F_{iSubband},'corr','circ');
                % Frequency domain
                %Y{iSubband} = real(ifftn(U.*conj(F_{iSubband})));
            end
            for iSubband = 1:nSubbands_
                coefs_((iSubband-1)*nPixels_+1:iSubband*nPixels_) = ...
                    reshape(Y{iSubband},[],1);
            end
            obj.coefs = coefs_;
        end
    end

    methods (Access = private)
       
        function F = filters_(obj)
            nLevels_ = obj.NumberOfLevels;
            K = obj.kernels;
            % iLevel == nLevels
            ufactor = 2^(nLevels_-1);
            kernelSize = 2^nLevels_;
            weight = 1/kernelSize;
            Ku = cellfun(@(x) ...
                cellfun(@(y) filterupsample1_(obj,y,ufactor),x,'UniformOutput',false),...
                K,'UniformOutput',false);
            F = cellfun(@(x) ...
                cellfun(@(y) y*weight,x,'UniformOutput',false),...
                Ku,'UniformOutput',false);
            % iLevel < nLevels
            for iLevel = nLevels_-1:-1:1
                ufactor = ufactor/2;
                kernelSize = kernelSize/2;
                weight = 1/kernelSize;
                Ku = cellfun(@(x) ...
                    cellfun(@(y) filterupsample1_(obj,y,ufactor),x,'UniformOutput',false),...
                    K,'UniformOutput',false);                
                for iSubband = 1:((nLevels_-iLevel)*7)+1
                    F{iSubband} = cellfun(@(x,y) conv(x,y),F{iSubband},Ku{1},'UniformOutput',false);
                end
                for idx = 2:8
                    iSubband = (nLevels_- iLevel)*7+idx;
                    F{iSubband} = cellfun(@(x) x*weight,Ku{idx},'UniformOutput',false);
                end
            end
        end

        function value = filterupsample1_(~,x,ufactor)
            value = upsample(x,ufactor);
            value = value(1:end-ufactor+1);
        end
        %{
        function F = filters_(obj)
            nLevels_ = obj.nLevels;
            F = cell(nLevels_*7+1,1);
            K = obj.kernels;
            % iLevel == nLevels
            ufactor = 2^(nLevels_-1);
            kernelSize = 2^nLevels_;
            weight = 1/(kernelSize^3);
            Ku = cellfun(@(x) filterupsample3_(obj,x,ufactor),K,'UniformOutput',false);
            for idx = 1:8
                F{idx} = Ku{idx}*weight;
            end
            % iLevel < nLevels
            for iLevel = nLevels_-1:-1:1
                ufactor = ufactor/2;
                kernelSize = kernelSize/2;
                weight = 1/(kernelSize^3);
                Ku = cellfun(@(x) filterupsample3_(obj,x,ufactor),K,'UniformOutput',false);
                for iSubband = 1:((nLevels_-iLevel)*7)+1
                    F{iSubband} = convn(F{iSubband},Ku{1}); 
                end
                for idx = 2:8
                    iSubband = (nLevels_- iLevel)*7+idx;
                    F{iSubband} = Ku{idx}*weight;
                end
            end
        end

        function value = filterupsample3_(~,x,ufactor)
            value = shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,...
                ufactor),1),...
                ufactor),1),...
                ufactor),1);
            value = value(1:end-ufactor+1,1:end-ufactor+1,1:end-ufactor+1);
        end
        %}
    end
end

