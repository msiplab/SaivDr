classdef UdHaarSynthesis3dSystem <  saivdr.dictionary.AbstSynthesisSystem  %#codegen
    %DicUdHaarRec3 Synthesis system for undecimated Haar transform
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
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Circular'});
    end
    
    properties (Logical)
        UseParallel = false
    end
    
    properties (Nontunable, Access = private)
        kernels
        nPixels
        nLevels
        nWorkers
        dim
    end
    
    methods
        function obj = UdHaarSynthesis3dSystem(varargin)
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
            s = saveObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj);
            s.nWorkers = obj.nWorkers;
            s.kernels = obj.kernels;
            s.nPixels = obj.nPixels;
            s.nLevels = obj.nLevels;
            s.dim = obj.dim;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.nWorkers = s.nWorkers;            
            obj.kernels = s.kernels;
            obj.nPixels = s.nPixels;
            obj.nLevels = s.nLevels;
            obj.dim = s.dim;
            loadObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj,s,wasLocked);
        end
        
        function setupImpl(obj, coefs, scales)
            if isa(coefs,'gpuArray')
                obj.UseGpu = true;
            end
            obj.dim = scales(1,:);
            obj.nPixels = prod(obj.dim);
            obj.nLevels = (size(scales,1)-1)/7;
            %
            if obj.UseParallel
                obj.nWorkers = Inf;
            else
                obj.nWorkers = 0;
            end
        end
        
        function resetImpl(~)
        end
        
        function y = stepImpl(obj, coefs, ~)
            if obj.UseGpu
                coefs = gpuArray(coefs);
            end
            % NOTE:
            % imfilter of R2017a has a bug for double precision array
            if strcmp(version('-release'),'2017a') && ...
                    isa(u,'double')
                warning(['IMFILTER of R2017a with CIRCULAR option has a bug for double precison array.' ...
                    ' Please visit https://jp.mathworks.com/support/bugreports/ and search #BugID: 1554862.' ])
            end
            ufactor = 2^(obj.nLevels-1);
            kernelSize = 2^obj.nLevels;
            weight = 1/(kernelSize^3);
            %
            nPixels_ = obj.nPixels;
            dim_ = obj.dim;
            U = cell(8,1);
            Y = cell(8,1);
            for idx = 1:8
                U{idx} = reshape(coefs((idx-1)*nPixels_+1:idx*nPixels_),dim_);
            end
            K = obj.kernels;
            parfor (idx = 1:8, obj.nWorkers) % TODO ポリフェーズ実現 or IntegralBoxFilter3実現
                %     Y{idx} = imfilter(U{idx},upsample3_(K{idx},ufactor),...
                %         'conv','circular')*weight;
                Y{idx} = upsmplfilter3_(obj,U{idx},K{idx},ufactor)*weight;
            end
            
            nLevels_ = obj.nLevels;
            iSubband = 8;
            for iLevel = 1:nLevels_
                if nLevels_-iLevel < 2
                    offset = [1 1 1];
                else
                    offset = [1 1 1]*2^(nLevels_-iLevel-1);
                end
                
                U{1} = circshift(...
                    Y{1} + Y{2} + Y{3} + Y{4} + Y{5} + Y{6} + Y{7} + Y{8}, offset);
                %
                ufactor = ufactor/2;
                kernelSize = kernelSize/2;
                weight = 1/(kernelSize^3);
                if iLevel < nLevels_
                    for idx=2:8
                        U{idx} = reshape(coefs((iSubband+idx-2)*nPixels_+1:(iSubband+idx-1)*nPixels_),dim_)*weight;
                    end
                    parfor idx = 1:8 % TODO ポリフェーズ実現 or IntegralBoxFilter3実現
                        Y{idx} = upsmplfilter3_(obj,U{idx},K{idx},ufactor);
                    end
                    %
                    iSubband = iSubband + 7;
                end
            end
            y = U{1};
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
                'conv','circular');
        end
        
    end
end

