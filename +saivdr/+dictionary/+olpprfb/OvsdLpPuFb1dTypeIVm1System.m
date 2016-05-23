classdef OvsdLpPuFb1dTypeIVm1System < ...
        saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem %#codegen
    %OVSDLPPUFBMDTYPEIVM1SYSTEM 2-D Type-I Oversampled LPPUFB with one VM
    %
    % SVN identifier:
    % $Id: OvsdLpPuFb1dTypeIVm1System.m 653 2015-02-04 05:21:08Z sho $
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
        omgsV0_
        omgsW_
        omgsU_
    end
    
    methods
        function obj = OvsdLpPuFb1dTypeIVm1System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem(...
                varargin{:});
            obj.omgsV0_ = OrthonormalMatrixGenerationSystem();
            obj.omgsW_ = OrthonormalMatrixGenerationSystem();
            obj.omgsU_ = OrthonormalMatrixGenerationSystem();
        end
    end
    
    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem(obj);
            s.omgsV0_ = matlab.System.saveObject(obj.omgsV0_);
            s.omgsW_ = matlab.System.saveObject(obj.omgsW_);
            s.omgsU_ = matlab.System.saveObject(obj.omgsU_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem(obj,s,wasLocked);
            obj.omgsV0_ = matlab.System.loadObject(s.omgsV0_);
            obj.omgsW_ = matlab.System.loadObject(s.omgsW_);
            obj.omgsU_ = matlab.System.loadObject(s.omgsU_);
        end        
        
        function updateParameterMatrixSet_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            nChs = obj.NumberOfChannels;
            hChs = nChs/2;
            angles = obj.Angles;
            mus    = obj.Mus;
            % No-DC-Leakage condition
            angles(1:hChs-1) = ...
                zeros(hChs-1,1);
            mus(1) = 1;
            % V0 with No-DC-Leakage condition
            mtx = step(obj.omgsV0_,angles(1:nChs*(nChs-1)/2),mus(1:nChs));
            step(obj.ParameterMatrixSet,mtx,uint32(1));
            
            obj.Angles = angles;
            obj.Mus = mus;
            
            pmMtxSet = obj.ParameterMatrixSet;
            angles = reshape(angles(nChs*(nChs-1)/2+1:end),[],obj.nStages-1);
            mus    = reshape(mus(nChs+1:end),[],obj.nStages-1);
            omgsW     = obj.omgsW_;
            omgsU     = obj.omgsU_;
            nAngs = hChs*(hChs-1)/2;
            nMus = hChs;
            for iParamMtx = uint32(1):obj.nStages-1
                mtx = step(omgsW,angles(1:nAngs,iParamMtx),mus(1:nMus,iParamMtx));
                step(pmMtxSet,mtx,3*iParamMtx-1);
                mtx = step(omgsU,angles(nAngs+1:2*nAngs,iParamMtx),mus(nMus+1:end,iParamMtx));
                step(pmMtxSet,mtx,3*iParamMtx);
                step(pmMtxSet,angles(2*nAngs+1:end,iParamMtx),3*iParamMtx+1);
            end
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = 0;
        end
    end
    
end
