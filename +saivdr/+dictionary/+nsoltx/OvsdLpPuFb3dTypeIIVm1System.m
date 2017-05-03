classdef OvsdLpPuFb3dTypeIIVm1System < ...
        saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dTypeIISystem %#codegen
    %OVSDLPPUFB2DTYPEIIVM1SYSTEM 3-D Type-II Oversampled LPPUFB with one VM
    %
    % SVN identifier:
    % $Id: OvsdLpPuFb3dTypeIIVm1System.m 683 2015-05-29 08:22:13Z sho $
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
   
    properties (Access = private, Nontunable)
        omgsW_
        omgsU_
        omfs_
    end
      
    methods        
        function obj = OvsdLpPuFb3dTypeIIVm1System(varargin)
            import saivdr.dictionary.utility.OrthonormalMatrixFactorizationSystem
            import saivdr.dictionary.utility.OrthonormalMatrixGenerationSystem
            obj = obj@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dTypeIISystem(...
                varargin{:});
            obj.omfs_  = OrthonormalMatrixFactorizationSystem();
            obj.omgsW_ = OrthonormalMatrixGenerationSystem();
            obj.omgsU_ = OrthonormalMatrixGenerationSystem();
        end
    end
    
    methods (Access = protected)
            
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dTypeIISystem(obj);
            s.omfs_  = matlab.System.saveObject(obj.omfs_);            
            s.omgsW_ = matlab.System.saveObject(obj.omgsW_);            
            s.omgsU_ = matlab.System.saveObject(obj.omgsU_);            
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dTypeIISystem(obj,s,wasLocked);
            obj.omfs_ = matlab.System.loadObject(s.omfs_);
            obj.omgsW_ = matlab.System.loadObject(s.omgsW_);            
            obj.omgsU_ = matlab.System.loadObject(s.omgsU_);            
        end        
        
        function updateParameterMatrixSet_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            nChs = obj.NumberOfChannels;
            angles = obj.Angles;
            mus    = obj.Mus;
            nAngsW = nChs(ChannelGroup.UPPER)*(nChs(ChannelGroup.UPPER)-1)/2;
            nMusW  = nChs(ChannelGroup.UPPER);
            % No-DC-Leackage condition
            W_ = eye(nChs(ChannelGroup.UPPER)); 
            omgsW = obj.omgsW_;
            omgsU = obj.omgsU_;
            pmMtxSet = obj.ParameterMatrixSet;
            for iParamMtx = uint32(1):obj.nStages-1
                % W
                mtx = step(omgsW,angles(1:nAngsW,iParamMtx),...
                    mus(1:nMusW,iParamMtx));                  
                step(pmMtxSet,mtx,2*iParamMtx-1);
                % U
                mtx = step(omgsU,angles(nAngsW+1:end,iParamMtx),...
                        mus(nMusW+1:end,iParamMtx));
                step(pmMtxSet,mtx,2*iParamMtx);

                W_ = step(pmMtxSet,[],2*iParamMtx-1)*W_;
            end
            [angles_,mus_] = step(obj.omfs_,W_.');
            angles(1:nChs(ChannelGroup.UPPER)-1,obj.nStages) = ...
                angles_(1:nChs(ChannelGroup.UPPER)-1);
            mus(1,obj.nStages) = mus_(1);
            % W
            mtx = step(omgsW,angles(1:nAngsW,obj.nStages),...
                mus(1:nMusW,obj.nStages));            
            step(pmMtxSet,mtx,2*obj.nStages-1); 
            % U
            mtx = step(omgsU,angles(nAngsW+1:end,obj.nStages),...
                mus(nMusW+1:end,obj.nStages));
            step(pmMtxSet,mtx,2*obj.nStages);
            %
            obj.Angles = angles;
            obj.Mus    = mus;
        end
        
    end
    
    methods (Access = protected, Static = true)
        function value = getDefaultPolyPhaseOrder_()
            value = [ 0 0 0 ];
        end
    end
end
