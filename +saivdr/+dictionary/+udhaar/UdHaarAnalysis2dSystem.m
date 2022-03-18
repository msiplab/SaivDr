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
        filters
    end

    methods
        function obj = UdHaarAnalysis2dSystem(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
            % 
            obj.kernels{1} = [ 1 1 ; 1 1 ]; % A
            obj.kernels{2} = [ 1 -1 ; 1 -1 ]; % H
            obj.kernels{3} = [ 1 1 ; -1 -1 ]; % V
            obj.kernels{4} = [ 1 -1 ; -1 1 ]; % D
        end
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj);
            s.kernels = obj.kernels;
            s.coefs = obj.coefs;
            s.nPixels = obj.nPixels;
            s.filters = obj.filters;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.kernels = s.kernels;
            obj.coefs = s.coefs;
            obj.nPixels = s.nPixels;
            obj.filters = s.filters;
            loadObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj,s,wasLocked); 
        end
        
        function setupImpl(obj,u)
            nLevels = obj.NumberOfLevels;
            obj.nPixels = numel(u);
            obj.coefs = zeros(1,(3*nLevels+1)*obj.nPixels,'like',u);
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
            nLevels_ = obj.NumberOfLevels;
            nPixels_ = obj.nPixels;
            coefs_ = obj.coefs;
            %
            F_ = obj.filters;
            scales = repmat(size(u),[3*nLevels_+1, 1]);   
            nSubbands_ = nLevels_*3 + 1;
            for iSubband = 1:nSubbands_
                coefs_((iSubband-1)*nPixels_+1:iSubband*nPixels_) = ...
                    reshape(imfilter(u,F_{iSubband},'corr','circ'),[],1);
            end
            obj.coefs = coefs_;
        end

    end
    
   
    methods (Access = private)
        
        function F = filters_(obj)
            nLevels_ = obj.NumberOfLevels;
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