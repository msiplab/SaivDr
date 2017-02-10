classdef CplxOvsdLpPuFb2dTypeIVm0System < saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dTypeISystem %#codegen
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
        initOmgs_
        propOmgs_
    end
    
    methods
        function obj = CplxOvsdLpPuFb2dTypeIVm0System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dTypeISystem(...
                varargin{:});
            obj.initOmgs_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            obj.propOmgs_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dTypeISystem(obj);
            s.initOmgs_ = matlab.System.saveObject(obj.initOmgs_);
            s.propOmgs_ = matlab.System.saveObject(obj.propOmgs_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.initOmgs_ = matlab.System.loadObject(s.initOmgs_);
            obj.propOmgs_ = matlab.System.loadObject(s.propOmgs_);
            loadObjectImpl@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dTypeISystem(obj,s,wasLocked);
        end
        
        function obj = updateParameterMatrixSet_(obj)
            nCh = sum(obj.NumberOfChannels);
            
            [initAngles, propAngles] = splitAngles_(obj);
            
            % initial matrix
            mtx = step(obj.initOmgs_,initAngles,obj.Mus(:,1));
            step(obj.ParameterMatrixSet,mtx,uint32(1));
            
            angles = reshape(propAngles,[],obj.nStages-1);
            mus    = obj.Mus(:,2:end);
            
            nParamMtxAngs = nCh*(nCh-2)/8;
            for iParamMtx = uint32(1):obj.nStages-1
                % W
                mtx = step(obj.propOmgs_,angles(1:nParamMtxAngs,iParamMtx),mus(1:nCh/2,iParamMtx));
                step(obj.ParameterMatrixSet,mtx,3*iParamMtx-1);
                
                % U
                mtx = step(obj.propOmgs_,angles(nParamMtxAngs+1:2*nParamMtxAngs,iParamMtx),mus(nCh/2+1:end,iParamMtx));
                step(obj.ParameterMatrixSet,mtx,3*iParamMtx);
                
                % angsB
                step(obj.ParameterMatrixSet,angles(2*nParamMtxAngs+1:end,iParamMtx),3*iParamMtx+1);
            end
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = [ 0 0 ];
        end
    end
    
end
