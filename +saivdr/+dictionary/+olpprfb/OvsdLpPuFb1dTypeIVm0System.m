classdef OvsdLpPuFb1dTypeIVm0System < ...
        saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem %#codegen
    %OVSDLPPUFBMDTYPEIVM0SYSTEM 2-D Type-I Oversapled LPPUFB without VM
    %
    % SVN identifier:
    % $Id: OvsdLpPuFb1dTypeIVm0System.m 653 2015-02-04 05:21:08Z sho $
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2014, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627    
    % 
    
    properties (Access = private)
        omgsW_
        omgsU_
        %TODO: angles
    end
    
    methods
        function obj = OvsdLpPuFb1dTypeIVm0System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem(...
                varargin{:});
            obj.omgsW_ = OrthonormalMatrixGenerationSystem();
            obj.omgsU_ = OrthonormalMatrixGenerationSystem();
            %TODO: angles
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem(obj);
            s.omgsW_ = matlab.System.saveObject(obj.omgsW_);
            s.omgsU_ = matlab.System.saveObject(obj.omgsU_);
            %TODO: angles
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.omgsW_ = matlab.System.loadObject(s.omgsW_);
            obj.omgsU_ = matlab.System.loadObject(s.omgsU_);
            %TODO: angles
            loadObjectImpl@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem(obj,s,wasLocked);
        end
        
        function obj = updateParameterMatrixSet_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            hChs = obj.NumberOfChannels/2;
            angles = obj.Angles;
            mus    = obj.Mus;
            nAngs = hChs*(hChs-1)/2;
            nMus = hChs;
            %pmMtxSt_ = obj.ParamteterMatrixSet;
            for iParamMtx = uint32(1):obj.nStages
                % W
                mtx = step(obj.omgsW_,angles(1:nAngs,iParamMtx),mus(1:nMus,iParamMtx));
                step(obj.ParameterMatrixSet,mtx,2*iParamMtx-1);
                % U
                mtx = step(obj.omgsU_,angles(nAngs+1:end,iParamMtx),mus(nMus+1:end,iParamMtx));
                step(obj.ParameterMatrixSet,mtx,2*iParamMtx);
            end
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = 0;
        end
    end
    
end
