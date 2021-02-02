classdef OvsdLpPuFb2dTypeIIVm0System < saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeIISystem
    %OVSDLPPUFB2DTYPEIIVM0SYSTEM 2-D Type-II Oversampled LPPUFB without VM
    %
    % SVN identifier:
    % $Id: OvsdLpPuFb2dTypeIIVm0System.m 683 2015-05-29 08:22:13Z sho $
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

    properties (Access = private)
        omgsW_
        omgsU_
    end
    
    methods
        function obj = OvsdLpPuFb2dTypeIIVm0System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeIISystem(...
                varargin{:});
            obj.omgsW_ = OrthonormalMatrixGenerationSystem();
            obj.omgsU_ = OrthonormalMatrixGenerationSystem();
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeIISystem(obj);
            s.omgsW_ = matlab.System.saveObject(obj.omgsW_);
            s.omgsU_ = matlab.System.saveObject(obj.omgsU_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.omgsW_ = matlab.System.loadObject(s.omgsW_);
            obj.omgsU_ = matlab.System.loadObject(s.omgsU_);
            loadObjectImpl@saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeIISystem(obj,s,wasLocked);
        end                
        
        function updateParameterMatrixSet_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            nChs = obj.NumberOfChannels;
            angles = obj.Angles;
            mus    = obj.Mus;            
            nAngsW = nChs(ChannelGroup.UPPER)*(nChs(ChannelGroup.UPPER)-1)/2;
            nMusW = nChs(ChannelGroup.UPPER);
            for iParamMtx = uint32(1):obj.nStages
                % W
                mtx = step(obj.omgsW_,angles(1:nAngsW,iParamMtx),...
                    mus(1:nMusW,iParamMtx));
                step(obj.ParameterMatrixSet,mtx,2*iParamMtx-1);
                % U
                mtx = step(obj.omgsU_,angles(nAngsW+1:end,iParamMtx),...
                        mus(nMusW+1:end,iParamMtx));
                step(obj.ParameterMatrixSet,mtx,2*iParamMtx);
            end
        end

    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = [ 0 0 ];
        end
    end
    
end
