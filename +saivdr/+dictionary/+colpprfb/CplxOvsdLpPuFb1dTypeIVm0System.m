classdef CplxOvsdLpPuFb1dTypeIVm0System < ...
        saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeISystem %#codegen
    %OVSDLPPUFBMDTYPEIVM0SYSTEM 2-D Type-I Oversapled LPPUFB without VM
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2014-2016, Shogo MURAMATSU
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
        omgsWU_
    end
    
    methods
        function obj = CplxOvsdLpPuFb1dTypeIVm0System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeISystem(...
                varargin{:});
            obj.omgsV0_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            obj.omgsWU_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeISystem(obj);
            s.omgsV0_ = matlab.System.saveObject(obj.omgsV0_);
            s.omgsWU_ = matlab.System.saveObject(obj.omgsWU_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.omgsV0_ = matlab.System.loadObject(s.omgsV0_);
            obj.omgsWU_ = matlab.System.loadObject(s.omgsWU_);
            loadObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeISystem(obj,s,wasLocked);
        end
        
        function obj = updateParameterMatrixSet_(obj)
            import saivdr.dictionary.cnsoltx.ChannelGroup
            nChs = obj.NumberOfChannels;
            
            % V0
            mtx = step(obj.omgsV0_,obj.Angles(1:nChs*(nChs-1)/2),obj.Mus(1:nChs));
            step(obj.ParameterMatrixSet,mtx,uint32(1));
            
            hChs = obj.NumberOfChannels/2;
            angles = reshape(obj.Angles(nChs*(nChs-1)/2+1:end),[],obj.nStages-1);
            mus    = reshape(obj.Mus(nChs+1:end),[],obj.nStages-1);
            nAngs = hChs*(hChs-1)/2;
            nMus = hChs;
            %TODO:
            for iParamMtx = uint32(1):obj.nStages-1
                % W
                mtx = step(obj.omgsWU_,angles(1:nAngs,iParamMtx),...
                    mus(1:nMus,iParamMtx));
                step(obj.ParameterMatrixSet,mtx,3*iParamMtx-1);
                % U
                mtx = step(obj.omgsWU_,angles(nAngs+1:2*nAngs,iParamMtx),...
                    mus(nMus+1:end,iParamMtx));
                step(obj.ParameterMatrixSet,mtx,3*iParamMtx+0);
                % angles_B
                step(obj.ParameterMatrixSet,angles(2*nAngs+1:end,iParamMtx),...
                    3*iParamMtx+1);
            end
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = 0;
        end
    end
    
end
