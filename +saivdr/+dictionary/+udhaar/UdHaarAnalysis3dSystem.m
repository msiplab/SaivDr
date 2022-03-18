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
            %
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
            obj.filters = filters_(obj);

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
            parfor (iSubband = 1:nSubbands_, obj.nWorkers)
                Y{iSubband} = imfilter(u,F_{iSubband},'corr','circ');
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

    end
end

