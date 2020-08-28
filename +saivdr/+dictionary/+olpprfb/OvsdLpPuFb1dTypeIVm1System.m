classdef OvsdLpPuFb1dTypeIVm1System < ...
        saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem %#codegen
    %OVSDLPPUFBMDTYPEIVM1SYSTEM 2-D Type-I Oversampled LPPUFB with one VM
    %
    % SVN identifier:
    % $Id: OvsdLpPuFb1dTypeIVm1System.m 653 2015-02-04 05:21:08Z sho $
    %
    % Requirements: MATLAB R2015b
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
    % http://msiplab.eng.niigata-u.ac.jp/    
    %      
    
    properties (Access = private)
        omgs_
    end
    
    methods
        function obj = OvsdLpPuFb1dTypeIVm1System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem(...
                varargin{:});
            obj.omgs_ = OrthonormalMatrixGenerationSystem();
        end
    end
    
    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem(obj);
            s.omgs_ = matlab.System.saveObject(obj.omgs_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem(obj,s,wasLocked);
            obj.omgs_ = matlab.System.loadObject(s.omgs_);
        end        
        
        function updateParameterMatrixSet_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            nChs = obj.NumberOfChannels;
            angles = obj.Angles;
            mus    = obj.Mus;
            % No-DC-Leackage condition
            angles(1:nChs(ChannelGroup.LOWER)-1,1) = ...
                zeros(nChs(ChannelGroup.LOWER)-1,1);
            mus(1,1) = 1;
            pmMtxSet = obj.ParameterMatrixSet;
            omgs     = obj.omgs_;
            for iParamMtx = uint32(1):obj.nStages+1
                mtx = step(omgs,angles(:,iParamMtx),mus(:,iParamMtx));
                step(pmMtxSet,mtx,iParamMtx);
            end
            obj.Angles = angles;
            obj.Mus = mus;
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = 0;
        end
    end
    
end
