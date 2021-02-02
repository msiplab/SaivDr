classdef NsoltFactory
    %NSOLTFACTORY Factory class of NSOLTs
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2020, Shogo MURAMATSU
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
    
    methods (Static = true)
        

        function value = createAnalysisSystem(varargin)
            import saivdr.dictionary.nsoltx.NsoltFactory
            if isa(varargin{1},...
                    'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem')
                value = NsoltFactory.createAnalysis2dSystem(varargin{:});
            elseif isa(varargin{1},...
                    'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem')
                value = NsoltFactory.createAnalysis3dSystem(varargin{:});
            else
                error('SaivDr: Invalid argumetns');
            end
        end
        
        function value = createSynthesisSystem(varargin)
            import saivdr.dictionary.nsoltx.NsoltFactory            
            if isa(varargin{1},...
                    'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem')
                value = NsoltFactory.createSynthesis2dSystem(varargin{:});
            elseif isa(varargin{1},...
                    'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem')
                value = NsoltFactory.createSynthesis3dSystem(varargin{:});
            else
                error('SaivDr: Invalid argumetns');
            end
        end        
                
        function value = createOvsdLpPuFb2dSystem(varargin)
            import saivdr.dictionary.nsoltx.*
            if nargin < 1
                value = OvsdLpPuFb2dTypeIVm1System();
            elseif isa(varargin{1},'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem')
                value = clone(varargin{1});
            else
                p = inputParser;
                addOptional(p,'DecimationFactor',[2 2]);
                addOptional(p,'NumberOfChannels',[]);
                addOptional(p,'NumberOfVanishingMoments',1);
                addOptional(p,'PolyPhaseOrder',[]);
                addOptional(p,'OutputMode',[]);
                parse(p,varargin{:});
                nDecs = p.Results.DecimationFactor;
                nChs = p.Results.NumberOfChannels;
                vm = p.Results.NumberOfVanishingMoments;
                if isempty(nChs)
                    nChs = [ceil(prod(nDecs)/2) floor(prod(nDecs)/2) ];
                elseif isscalar(nChs)
                    nChs = [ceil(nChs/2) floor(nChs/2) ];
                end
                if nChs(ChannelGroup.UPPER) == nChs(ChannelGroup.LOWER)
                    type = 1;
                else
                    type = 2;
                end
                % 
                newidx = 1;
                oldidx = 1;
                while oldidx < nargin+1
                    if ischar(varargin{oldidx}) && ...
                            strcmp(varargin{oldidx},'NumberOfVanishingMoments')
                        oldidx = oldidx + 1;
                    else
                        argin{newidx} = varargin{oldidx};
                        newidx = newidx + 1;
                    end
                    oldidx = oldidx + 1;
                end
                if vm == 0
                    if type == 1
                        value = OvsdLpPuFb2dTypeIVm0System(argin{:});
                    else
                        value = OvsdLpPuFb2dTypeIIVm0System(argin{:});
                    end
                elseif vm == 1
                    if type == 1
                        value = OvsdLpPuFb2dTypeIVm1System(argin{:});
                    else
                        value = OvsdLpPuFb2dTypeIIVm1System(argin{:});
                    end
                end
            end
        end
        
        function value = createOvsdLpPuFb3dSystem(varargin)
            import saivdr.dictionary.nsoltx.*
            if nargin < 1
                value = OvsdLpPuFb3dTypeIVm1System();
            elseif isa(varargin{1},'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem')
                value = clone(varargin{1});
            else
                p = inputParser;
                addOptional(p,'DecimationFactor',[2 2 2]);
                addOptional(p,'NumberOfChannels',[]);
                addOptional(p,'NumberOfVanishingMoments',1);
                addOptional(p,'PolyPhaseOrder',[]);
                addOptional(p,'OutputMode',[]);
                parse(p,varargin{:});
                nDecs = p.Results.DecimationFactor;
                nChs = p.Results.NumberOfChannels;
                vm = p.Results.NumberOfVanishingMoments;
                if isempty(nChs)
                    nChs = [ceil(prod(nDecs)/2) floor(prod(nDecs)/2) ];
                elseif isscalar(nChs)
                    nChs = [ceil(nChs/2) floor(nChs/2) ];
                end
                if nChs(ChannelGroup.UPPER) == nChs(ChannelGroup.LOWER)
                    type = 1;
                else
                    type = 2;
                end
                % 
                newidx = 1;
                oldidx = 1;
                while oldidx < nargin+1
                    if ischar(varargin{oldidx}) && ...
                            strcmp(varargin{oldidx},'NumberOfVanishingMoments')
                        oldidx = oldidx + 1;
                    else
                        argin{newidx} = varargin{oldidx};
                        newidx = newidx + 1;
                    end
                    oldidx = oldidx + 1;
                end
                if vm == 0
                    if type == 1
                        value = OvsdLpPuFb3dTypeIVm0System(argin{:});
                    else
                        value = OvsdLpPuFb3dTypeIIVm0System(argin{:});
                    end
                elseif vm == 1
                    if type == 1
                        value = OvsdLpPuFb3dTypeIVm1System(argin{:});
                    else
                        value = OvsdLpPuFb3dTypeIIVm1System(argin{:});
                    end
                end
            end
        end        
        
            
        function value = createAnalysis2dSystem(varargin)
            import saivdr.dictionary.nsoltx.NsoltAnalysis2dSystem
            import saivdr.dictionary.nsoltx.NsoltFactory
            if nargin < 1 
                value = NsoltAnalysis2dSystem();
            elseif isa(varargin{1},...
                    'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem')
                value = NsoltAnalysis2dSystem('LpPuFb2d',varargin{:});
            else
                error('SaivDr: Invalid arguments');
            end
        end
        
        function value = createSynthesis2dSystem(varargin)
            import saivdr.dictionary.nsoltx.NsoltSynthesis2dSystem
            import saivdr.dictionary.nsoltx.NsoltFactory
            if nargin < 1
                value = NsoltSynthesis2dSystem();
            elseif isa(varargin{1},...
                    'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem')
                value = NsoltSynthesis2dSystem('LpPuFb2d',varargin{:});
            else
                error('SaivDr: Invalid arguments');
            end
        end        
        
        function value = createAnalysis3dSystem(varargin)
            import saivdr.dictionary.nsoltx.NsoltAnalysis3dSystem
            import saivdr.dictionary.nsoltx.NsoltFactory
            if nargin < 1 
                value = NsoltAnalysis3dSystem();
            elseif isa(varargin{1},...
                    'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem')
                value = NsoltAnalysis3dSystem('LpPuFb3d',varargin{:});
            else
                error('SaivDr: Invalid arguments');
            end
        end
        
        function value = createSynthesis3dSystem(varargin)
            import saivdr.dictionary.nsoltx.NsoltSynthesis3dSystem
            import saivdr.dictionary.nsoltx.NsoltFactory
            if nargin < 1
                value = NsoltSynthesis3dSystem();
            elseif isa(varargin{1},...
                    'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem')
                value = NsoltSynthesis3dSystem('LpPuFb3d',varargin{:});
            else
                error('SaivDr: Invalid arguments');
            end
        end  
        
    end
    
end

