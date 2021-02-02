classdef MixtureOfUnitarySynthesisSystem < saivdr.dictionary.AbstSynthesisSystem %~#codegen
    %MIXTUREOFUNITARYSYNTHESISSYSTEM Mixture of unitary synthesis system 
    %
    % SVN identifier:
    % $Id: MixtureOfUnitarySynthesisSystem.m 683 2015-05-29 08:22:13Z sho $
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
        UnitarySynthesizerSet
    end
    
    properties (Access = private)
        nSynthesizers
        normalizationFactor
    end
    
    methods
        
        function obj = MixtureOfUnitarySynthesisSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            obj.nSynthesizers = length(obj.UnitarySynthesizerSet);
            obj.normalizationFactor = 1/sqrt(obj.nSynthesizers);
            obj.FrameBound = 1;
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            
            % Save the child System objects
            for iDic = 1:obj.nSynthesizers
                s.UnitarySynthesizerSet{iDic} = ...
                    matlab.System.saveObject(obj.UnitarySynthesizerSet{iDic});
            end
            
            % Save the protected & private properties
            s.nSynthesizers = obj.nSynthesizers;
            s.normalizationFactor = obj.normalizationFactor;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load protected and private properties
            obj.nSynthesizers = s.nSynthesizers;
            obj.normalizationFactor = s.normalizationFactor;
            
            % Call base class method to load public properties
            loadObjectImpl@matlab.System(obj,s,wasLocked);
            %
            for iDic = 1:obj.nSynthesizers
                obj.UnitarySynthesizerSet{iDic} = ...
                    matlab.System.loadObject(s.UnitarySynthesizerSet{iDic});
            end
        end
        
        function recImg = stepImpl(obj,coefs,scales)
           
            recImg = 0;
            iCoef  = 1;
            subIdxIn  = 1;
            subScales = cell(obj.nSynthesizers,1);
            for iDic = 1:obj.nSynthesizers-1
                subIdxOut = 1;
                while ~(scales(subIdxIn,1) == -1 && scales(subIdxIn,2) == -1)
                    subScales{iDic}(subIdxOut,:) = scales(subIdxIn,:);
                    subIdxOut = subIdxOut + 1;
                    subIdxIn  = subIdxIn  + 1;
                end
                subIdxIn  = subIdxIn  + 1;
            end
            iDic = obj.nSynthesizers;
            subScales{iDic} = scales(subIdxIn:end,:);
            for iDic = 1:obj.nSynthesizers
                subSynthesizer = obj.UnitarySynthesizerSet{iDic};
                nCoefs = subScales{iDic}(:,1).'*subScales{iDic}(:,2);
                subCoefs = coefs(iCoef:iCoef+nCoefs-1);
                recImg = recImg + ...
                    step(subSynthesizer,subCoefs,subScales{iDic});
                iCoef  = iCoef + nCoefs;
            end
            recImg = obj.normalizationFactor * recImg;

        end
        
    end
    
end
