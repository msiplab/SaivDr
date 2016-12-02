classdef OvsdLpPuFb2dTypeIVm1System < saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem %#codegen
    %OVSDLPPUFBMDTYPEIVM1SYSTEM 2-D Type-I Oversampled LPPUFB with one VM
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627    
    %      
    
    properties (Access = private)
        initOmgs_
        propOmgs_
        iniOmfs_
        propOmfs_
    end
    
    methods
        function obj = OvsdLpPuFb2dTypeIVm1System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            import saivdr.dictionary.utility.OrthonormalMatrixFactorizationSystem
            obj = obj@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem(...
                varargin{:});
            obj.initOmgs_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            obj.propOmgs_ = OrthonormalMatrixGenerationSystem('OrderOfProduction','Ascending');
            %obj.initOmfs_ = OrthonormalMatrixFactorizationSystem('OrderOfProduction','Ascending');
            obj.propOmfs_ = OrthonormalMatrixFactorizationSystem('OrderOfProduction','Ascending');
        end
    end
    
    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem(obj);
            s.initOmgs_ = matlab.System.saveObject(obj.initOmgs_);
            s.propOmgs_ = matlab.System.saveObject(obj.propOmgs_);
            %s.initOmfs_ = matlab.System.saveObject(obj.initOmfs_);
            s.propOmfs_ = matlab.System.saveObject(obj.propOmfs_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem(obj,s,wasLocked);
            obj.initOmgs_ = matlab.System.loadObject(s.initOmgs_);
            obj.propOmgs_ = matlab.System.loadObject(s.propOmgs_);
            %obj.initOmfs_ = matlab.System.loadObject(s.initOmfs_);
            obj.propOmfs_ = matlab.System.loadObject(s.propOmfs_);
        end        
        
        function updateParameterMatrixSet_(obj)
            %import saivdr.dictionary.nsoltx.ChannelGroup
            nch = sum(obj.NumberOfChannels);

            [initAngles, propAngles] = splitAngles_(obj);
            
            angles = reshape(propAngles,[],obj.nStages-1);
            mus    = obj.Mus(:,2:end);
            
            nParamMtxAngs = nch*(nch-2)/8;
%             % No-DC-Leackage condition
%             angles(1:nChs(ChannelGroup.LOWER)-1,1) = ...
%                 zeros(nChs(ChannelGroup.LOWER)-1,1);
%             mus(1,1) = 1;

            W_ = eye(nch/2);
            
            pmMtxSet = obj.ParameterMatrixSet;
            omgs     = obj.propOmgs_;
            for iParamMtx = uint32(1):obj.nStages-1
                %TODO: No-DC-Leakage condition??????????
                % W
                mtx = step(omgs,angles(1:nParamMtxAngs,iParamMtx),mus(1:nch/2,iParamMtx));
                step(pmMtxSet,mtx,3*iParamMtx-1);
                W_ = mtx*W_;
                
                % U
                mtx = step(omgs,angles(nParamMtxAngs+1:2*nParamMtxAngs,iParamMtx),mus(nch/2+1:end,iParamMtx));
                step(pmMtxSet,mtx,3*iParamMtx);
                
                % angsB
                step(pmMtxSet,angles(2*nParamMtxAngs+1:end,iParamMtx),3*iParamMtx+1);
            end
            
            % initial matrix
            [angles_,~] = step(obj.propOmfs_,W_.');
            initAngles(1:nch/2-1) = angles_(1:nch/2-1);
            initAngles(nch/2:nch-1) = zeros(1,nch/2);
            mtx = step(obj.initOmgs_,initAngles,obj.Mus(:,1));
            step(obj.ParameterMatrixSet,mtx,uint32(1));
            
            angles = [initAngles; angles(:)];
            mus = [obj.Mus(:,1), mus];
            
            obj.Angles = angles;
            obj.Mus = mus;
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = [ 0 0 ];
        end
    end
    
end
