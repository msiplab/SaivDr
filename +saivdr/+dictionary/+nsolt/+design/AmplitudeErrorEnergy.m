classdef AmplitudeErrorEnergy < matlab.System %#codegen
    %AMPLITUDEERRORENERGY Amplitude error energy
    %
    % This class evaluates the cost of given filterbanks in terms of the 
    % error energy between those filters and given ideal frequency 
    % specifications.
    %
    % SVN identifier:
    % $Id: AmplitudeErrorEnergy.m 683 2015-05-29 08:22:13Z sho $
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
    properties  (Nontunable)
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
        
        function obj = AmplitudeErrorEnergy(varargin)
            setProperties(obj,nargin,varargin{:});
        end
        
    end
    
    methods (Access = protected)
        
        function setupImpl(obj,varargin)
            obj.nSpecs = size(obj.AmplitudeSpecs,3);
        end
        
        function value = stepImpl(obj,varargin)
            if strcmp(obj.EvaluationMode,'All')
                filters_ = varargin{1};
                value = getCost_(obj,filters_);
            else
                filter_ = varargin{1};
                idx_ = varargin{2};
                value = getCostAt_(obj,filter_,idx_);
            end
        end
        
        function N = getNumInputsImpl(obj)
            if strcmp(obj.EvaluationMode,'All')
                N = 1;
            else
                N = 2;
            end
        end
        
        function N = getNumOutputsImpl(~)
            N = 1;
        end
        
    end
    
    methods (Access = private)
        
        
        function value = getCost_(obj,filters_)
            value = 0.0;
            if isa(filters_,'saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dSystem')
                lppufb = clone(filters_);
                if ~strcmp(get(lppufb,'OutputMode'),'AnalysisFilters')
                    warning('OutputMode of OvsdLpPuFb2d is recommended to be AnalysisFilters');
                    release(lppufb)
                    set(lppufb,'OutputMode','AnalysisFilters')
                end                
                filters_ = step(lppufb,[],[]);
            end
            for iSpec=1:obj.nSpecs
                value = value + ...
                    getCostAt_(obj,filters_(:,:,iSpec),iSpec);
            end
        end
        
        function value = getCostAt_(obj,filter,iSpec)
            nRows = size(obj.AmplitudeSpecs,1);
            nCols = size(obj.AmplitudeSpecs,2);
            ampres = fftshift(abs(fft2(filter,nRows,nCols)));
            diff = ampres-obj.AmplitudeSpecs(:,:,iSpec);
            value = diff(:).'*diff(:);
        end
        
    end
    
end
