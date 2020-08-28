classdef UdHaarAnalysis3dSystem < saivdr.dictionary.AbstAnalysisSystem %#codegen
    %UDHAARANALYSIS3DSYSTEM Analysis system for undecimated Haar transform
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2018-2020, Shogo MURAMATSU
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
            K{1}(:,:,1) = double([ 1 -1 ; -1 1 ]); % DD
            K{1}(:,:,2) = double(-[ 1 -1 ; -1 1 ]);
            K{2}(:,:,1) = double([ 1 1 ; -1 -1 ]); % VD
            K{2}(:,:,2) = double(-[ 1 1 ; -1 -1 ]);
            K{3}(:,:,1) = double([ 1 -1 ; 1 -1 ]); % HD
            K{3}(:,:,2) = double(-[ 1 -1 ; 1 -1 ]);
            K{4}(:,:,1) = double([ 1 1 ; 1 1 ]);   % AD
            K{4}(:,:,2) = double(-[ 1 1 ; 1 1 ]);
            %
            K{5}(:,:,1) = double([ 1 -1 ; -1 1 ]); % DA
            K{5}(:,:,2) = double([ 1 -1 ; -1 1 ]);
            K{6}(:,:,1) = double([ 1 1 ; -1 -1 ]); % VA
            K{6}(:,:,2) = double([ 1 1 ; -1 -1 ]);
            K{7}(:,:,1) = double([ 1 -1 ; 1 -1 ]); % HA
            K{7}(:,:,2) = double([ 1 -1 ; 1 -1 ]);
            K{8}(:,:,1) = double([ 1 1 ; 1 1 ]);   % AA
            K{8}(:,:,2) = double([ 1 1 ; 1 1 ]);
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
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.nWorkers = s.nWorkers;            
            obj.kernels = s.kernels;
            obj.coefs = s.coefs;
            obj.nPixels = s.nPixels;
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
            nPixels_ = obj.nPixels;
            coefs_ = obj.coefs;
            nLevels = obj.NumberOfLevels;
            K = obj.kernels;
            scales = repmat(size(u),[7*nLevels+1, 1]);
            yaa = u;
            
            Y = cell(8,1);
            %
            ufactor = uint32(1);
            iSubband = uint32(7*nLevels+1);
            weight = double(0);
            for iLevel = 1:nLevels
                kernelSize = 2^iLevel;
                weight = 1/(kernelSize^3);
                if iLevel < 2
                    offset = [0 0 0]; % 1
                else
                    offset = -[1 1 1]*(2^(iLevel-2)-1);
                end
                %
                parfor (idx = 1:8, obj.nWorkers)
                    Y{idx} = circshift(upsmplfilter3_(obj,yaa,K{idx},ufactor),offset);
                end
                for idx = 1:7
                    ytmp = Y{idx};
                    coefs_((iSubband-idx)*nPixels_+1:(iSubband-idx+1)*nPixels_) = ...
                        ytmp(:).'*weight;
                end
                iSubband = iSubband - 7;
                ufactor = ufactor*2;
                yaa = Y{8};
            end
            coefs_(1:nPixels_) = yaa(:).'*weight;
            obj.coefs = coefs_;
        end
    end


    methods (Access = private)
        
        function value = upsmplfilter3_(~,u,x,ufactor)
            value = imfilter(u,...
                shiftdim(upsample(...
                shiftdim(upsample(...
                shiftdim(upsample(x,...
                ufactor),1),...
                ufactor),1),...
                ufactor),1),...
                'corr','circular');
        end
        
    end
end

