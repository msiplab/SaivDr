classdef CnsoltFactory
    %NSOLTFACTORY Factory class of NSOLTs
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

    methods (Static = true)

        function value = createCplxOvsdLpPuFb2dSystem(varargin)
            import saivdr.dictionary.cnsoltx.*
            if nargin < 1
                value = CplxOvsdLpPuFb2dTypeIVm1System();
            elseif isa(varargin{1},'saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dSystem')
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
                        value = CplxOvsdLpPuFb2dTypeIVm0System(argin{:});
                    else
                        value = CplxOvsdLpPuFb2dTypeIIVm0System(argin{:});
                    end
                elseif vm == 1
                    if type == 1
                        value = CplxOvsdLpPuFb2dTypeIVm1System(argin{:});
                    else
                        value = CplxOvsdLpPuFb2dTypeIIVm1System(argin{:});
                    end
                end
            end
        end

        function value = createCplxOvsdLpPuFb3dSystem(varargin)
            import saivdr.dictionary.cnsoltx.*
            if nargin < 1
                value = CplxOvsdLpPuFb3dTypeIVm1System();
            elseif isa(varargin{1},'saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dSystem')
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
                        value = CplxOvsdLpPuFb3dTypeIVm0System(argin{:});
                    else
                        value = CplxOvsdLpPuFb3dTypeIIVm0System(argin{:});
                    end
                elseif vm == 1
                    if type == 1
                        value = CplxOvsdLpPuFb3dTypeIVm1System(argin{:});
                    else
                        value = CplxOvsdLpPuFb3dTypeIIVm1System(argin{:});
                    end
                end
            end
        end

        function value = createAnalysis2dSystem(varargin)
            import saivdr.dictionary.cnsoltx.CnsoltAnalysis2dSystem
            import saivdr.dictionary.cnsoltx.CnsoltFactory
            if nargin < 1
                value = CnsoltAnalysis2dSystem();
            elseif isa(varargin{1},...
                    'saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dSystem')
                value = CnsoltAnalysis2dSystem('LpPuFb2d',varargin{:});
            else
                error('SaivDr: Invalid arguments');
            end
        end

        function value = createSynthesis2dSystem(varargin)
            import saivdr.dictionary.cnsoltx.CnsoltSynthesis2dSystem
            import saivdr.dictionary.cnsoltx.CnsoltFactory
            if nargin < 1
                value = CnsoltSynthesis2dSystem();
            elseif isa(varargin{1},...
                    'saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb2dSystem')
                value = CnsoltSynthesis2dSystem('LpPuFb2d',varargin{:});
            else
                error('SaivDr: Invalid arguments');
            end
        end

        function value = createAnalysis3dSystem(varargin)
            import saivdr.dictionary.cnsoltx.CnsoltAnalysis3dSystem
            import saivdr.dictionary.cnsoltx.CnsoltFactory
            if nargin < 1
                value = CnsoltAnalysis3dSystem();
            elseif isa(varargin{1},...
                    'saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dSystem')
                value = CnsoltAnalysis3dSystem('LpPuFb3d',varargin{:});
            else
                error('SaivDr: Invalid arguments');
            end
        end

        function value = createSynthesis3dSystem(varargin)
            import saivdr.dictionary.cnsoltx.CnsoltSynthesis3dSystem
            import saivdr.dictionary.cnsoltx.CnsoltFactory
            if nargin < 1
                value = CnsoltSynthesis3dSystem();
            elseif isa(varargin{1},...
                    'saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dSystem')
                value = CnsoltSynthesis3dSystem('LpPuFb3d',varargin{:});
            else
                error('SaivDr: Invalid arguments');
            end
        end

    end

end
