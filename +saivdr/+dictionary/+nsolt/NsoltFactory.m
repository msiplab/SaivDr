classdef NsoltFactory
    %NSOLTFACTORY Factory class of NSOLTs
    %
    % SVN identifier:
    % $Id: NsoltFactory.m 683 2015-05-29 08:22:13Z sho $
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
    
    methods (Static = true)
        
        function value = createOvsdLpPuFb2dSystem(varargin)
            import saivdr.dictionary.nsolt.*
            if nargin < 1
                value = OvsdLpPuFb2dTypeIVm1System();
            elseif isa(varargin{1},'saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dSystem')
                value = varargin{1};
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
                        argin = cell(0);
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
            
        function value = createAnalysisSystem(varargin)
            import saivdr.dictionary.nsolt.TypeIAnalysisSystem
            import saivdr.dictionary.nsolt.TypeIIAnalysisSystem
            import saivdr.dictionary.nsolt.NsoltFactory
            if nargin < 1 
                value = TypeIAnalysisSystem();
            elseif isa(varargin{1},...
                    'saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeISystem')
                value = TypeIAnalysisSystem('LpPuFb2d',varargin{:});
            elseif isa(varargin{1},...
                    'saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeIISystem')
                value = TypeIIAnalysisSystem('LpPuFb2d',varargin{:});
            else
                error('SaivDr: Invalid arguments');
            end
        end
        
        function value = createSynthesisSystem(varargin)
            import saivdr.dictionary.nsolt.TypeISynthesisSystem
            import saivdr.dictionary.nsolt.TypeIISynthesisSystem
            import saivdr.dictionary.nsolt.NsoltFactory
            if nargin < 1
                value = TypeISynthesisSystem();
            elseif isa(varargin{1},...
                    'saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeISystem')
                value = TypeISynthesisSystem('LpPuFb2d',varargin{:});
            elseif isa(varargin{1},...
                    'saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeIISystem')
                value = TypeIISynthesisSystem('LpPuFb2d',varargin{:});
            else
                error('SaivDr: Invalid arguments');
            end
        end        
    end
    
end
