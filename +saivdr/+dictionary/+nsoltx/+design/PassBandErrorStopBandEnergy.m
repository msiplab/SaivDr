classdef PassBandErrorStopBandEnergy < matlab.System %#codegen
    %PASSBANDERRORSTOPBANDENERGY Passband error and stopband energy
    %
    % This class evaluates the cost of given filterbanks in terms of the 
    % error energy between those filters and given ideal frequency 
    % specifications.
    %
    % SVN identifier:
    % $Id: PassBandErrorStopBandEnergy.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
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
    % http://msiplab.eng.niigata-u.ac.jp/    
    %
    
    properties (Nontunable)
        AmplitudeSpecs
        EvaluationMode = 'All'
    end
    
    properties(Hidden,Transient)
        EvaluationModeSet = ...
            matlab.system.StringSet({'All', 'Individual'});
    end
    
    properties (GetAccess = public, SetAccess = protected)
        nSpecs
    end
    
    methods
        
        function obj = PassBandErrorStopBandEnergy(varargin)
            setProperties(obj,nargin,varargin{:});
        end
        
    end
    
    methods (Access = protected)
        
        function setupImpl(obj,varargin)
            obj.nSpecs = size(obj.AmplitudeSpecs,3);
        end
        
        function value = stepImpl(obj,varargin)
            filters_ = varargin{1};
            if strcmp(obj.EvaluationMode,'All')
                value = getCost_(obj,filters_);
            else
                idx_ = varargin{2};
                pgain_ = varargin{3};
                value = getCostAt_(obj,filters_,idx_,pgain_);
            end
        end
        
        function N = getNumInputsImpl(obj)
            if strcmp(obj.EvaluationMode,'All')
                N = 1;
            else
                N = 3;
            end
        end
        
        function N = getNumOutputsImpl(~)
            N = 1;
        end
        
    end
    
    methods (Access = private)
        
        function value = getCost_(obj,filters_)
            value = 0.0;
            pgain = [];
            if isa(filters_,'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem')
                lppufb = clone(filters_);
                if ~strcmp(get(lppufb,'OutputMode'),'AnalysisFilters')
                    warning('OutputMode of OvsdLpPuFb2d is recommended to be AnalysisFilters');
                    release(lppufb)
                    set(lppufb,'OutputMode','AnalysisFilters')
                end
                filters_ = step(lppufb,[],[]);
                pgain = sqrt(prod(get(lppufb,'DecimationFactor')));
            end
            for iSpec=1:obj.nSpecs
                value = value + ...
                    getCostAt_(obj,filters_(:,:,iSpec),iSpec,pgain);
            end
        end
        
        function value = getCostAt_(obj,filter,iSpec,pgain)
            if nnz(obj.AmplitudeSpecs) == 0
                value = 0.0;
            else
                nRows = size(obj.AmplitudeSpecs,1);
                nCols = size(obj.AmplitudeSpecs,2);
                ampres = fftshift(abs(fft2(filter,nRows,nCols)));
                if isempty(pgain)
                    targetValue = max(ampres(:));
                else
                    targetValue = pgain;
                end
                specPass = obj.AmplitudeSpecs(:,:,iSpec)>0;
                specStop = obj.AmplitudeSpecs(:,:,iSpec)<0;
                sqrderr = ((targetValue-ampres).*specPass ...
                    + (ampres.*specStop)).^2;
                value = sum(sqrderr(:));
            end
        end
        
    end
end
