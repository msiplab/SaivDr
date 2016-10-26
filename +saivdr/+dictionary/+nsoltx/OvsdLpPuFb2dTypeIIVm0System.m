classdef OvsdLpPuFb2dTypeIIVm0System < saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeIISystem
    %OVSDLPPUFBMDTYPEIIVM0SYSTEM M-D Type-II Oversampled LPPUFB without VM
    %
    % SVN identifier:
    % $Id: OvsdLpPuFb2dTypeIIVm0System.m 683 2015-05-29 08:22:13Z sho $
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
        omgsV0_
        omgsE_
        omgsO_
    end
    
    methods
        function obj = OvsdLpPuFb2dTypeIIVm0System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeIISystem(...
                varargin{:});
            obj.omgsV0_ = OrthonormalMatrixGenerationSystem();
            obj.omgsE_ = OrthonormalMatrixGenerationSystem();
            obj.omgsO_ = OrthonormalMatrixGenerationSystem();
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeIISystem(obj);
            s.omgsV0_ = matlab.System.saveObject(obj.omgsV0_);
            s.omgsE_ = matlab.System.saveObject(obj.omgsE_);
            s.omgsO_ = matlab.System.saveObject(obj.omgsO_);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.omgsV0_ = matlab.System.loadObject(s.omgsV0_);
            obj.omgsE_ = matlab.System.loadObject(s.omgsE_);
            obj.omgsO_ = matlab.System.loadObject(s.omgsO_);
            loadObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeIISystem(obj,s,wasLocked);
        end                
        
        function updateParameterMatrixSet_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            nCh = obj.NumberOfChannels;
            
            [initAngles, propAngles] = splitAngles_(obj);
            
            pmMtxSt_ = obj.ParameterMatrixSet;
            
            mtx = step(obj.omgsV0_,initAngles, obj.Mus(1:nCh));
            step(pmMtxSt_,mtx,uint32(1));
            
            angles = reshape(propAngles,[],obj.nStages-1);
            mus    = reshape(obj.Mus(nCh+1:end),[],obj.nStages-1);            
            nAngsW = floor(nCh/2)*(floor(nCh/2)-1)/2;
            nAngsHW = ceil(nCh/2)*(ceil(nCh/2)-1)/2;
            nAngsB = floor(nCh/4);
            nMusW = floor(nCh/2);
            nMusHW = ceil(nCh/2);
            omgsE = obj.omgsE_;
            omgsO = obj.omgsO_;
            for iParamMtx = uint32(1):obj.nStages-1
                % TODO: ???????????????t?@?N?^?????O????
                % W
                mtx = step(omgsE,angles(1:nAngsW,iParamMtx),...
                    mus(1:nMusW,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx-4);
                % U
                mtx = step(omgsE,angles(nAngsW+1:2*nAngsW,iParamMtx),...
                        mus(nMusW+1:2*nMusW,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx-3);
                
                % angsB1
                step(pmMtxSt_,angles(2*nAngsW+1:2*nAngsW+nAngsB,iParamMtx),6*iParamMtx-2);
                
                % HW
                mtx = step(omgsO,angles(2*nAngsW+nAngsB+1:2*nAngsW+nAngsB+nAngsHW,iParamMtx),...
                    mus(2*nMusW+1:2*nMusW+nMusHW,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx-1);
                % HU
                mtx = step(omgsO,angles(2*nAngsW+nAngsB+nAngsHW+1:2*nAngsW+nAngsB+2*nAngsHW,iParamMtx),...
                        mus(2*nMusW+nMusHW+1:end,iParamMtx));
                step(pmMtxSt_,mtx,6*iParamMtx);
                
                % angsB2
                step(pmMtxSt_,angles(2*nAngsW+1:2*nAngsW+nAngsB,iParamMtx),6*iParamMtx+1);
            end
        end

    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = [ 0 0 ];
        end
    end
    
end
