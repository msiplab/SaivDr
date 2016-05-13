classdef OLpPrFbFactory
    %OLpPrFbFactory Factory class of NSOLTs
    %
    % SVN identifier:
    % $Id: OLpPrFbFactory.m 657 2015-03-17 00:45:15Z sho $
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

    methods (Static = true)

        function value = createOvsdLpPuFb1dSystem(varargin)
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.olpprfb.*
            if nargin < 1
                value = OvsdLpPuFb1dTypeIVm1System();
            elseif isa(varargin{1},'saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dSystem')
                value = varargin{1};
            else
                p = inputParser;
                addOptional(p,'DecimationFactor',4);
                addOptional(p,'NumberOfChannels',[]);
                addOptional(p,'NumberOfVanishingMoments',1);
                addOptional(p,'PolyPhaseOrder',[]);
                addOptional(p,'OutputMode',[]);
                parse(p,varargin{:});
                nDec = p.Results.DecimationFactor;
                nChs = p.Results.NumberOfChannels;
                vm = p.Results.NumberOfVanishingMoments;
                if isempty(nChs)
                    nChs = nDec;
                elseif isvector(nChs)
                    nChs = sum(nChs);
                end
                if rem(nChs,2) == 0
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
                        value = OvsdLpPuFb1dTypeIVm0System(argin{:});
                    else
                        value = OvsdLpPuFb1dTypeIIVm0System(argin{:});
                    end
                elseif vm == 1
                    if type == 1
                        value = OvsdLpPuFb1dTypeIVm1System(argin{:});
                    else
                        value = OvsdLpPuFb1dTypeIIVm1System(argin{:});
                    end
                end
            end
        end

        function value = createAnalysis1dSystem(varargin)
            import saivdr.dictionary.olpprfb.OLpPuFbAnalysis1dSystem
            import saivdr.dictionary.olpprfb.OLpPrFbFactory
            if nargin < 1
                value = OLpPuFbAnalysis1dSystem();
            elseif isa(varargin{1},...
                    'saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dSystem')
                value = OLpPuFbAnalysis1dSystem('LpPuFb1d',varargin{:});
            else
                error('SaivDr: Invalid arguments');
            end
        end

        function value = createSynthesis1dSystem(varargin)
            import saivdr.dictionary.olpprfb.OLpPuFbSynthesis1dSystem
            import saivdr.dictionary.olpprfb.OLpPrFbFactory
            if nargin < 1
                value = OLpPuFbSynthesis1dSystem();
            elseif isa(varargin{1},...
                    'saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dSystem')
                value = OLpPuFbSynthesis1dSystem('LpPuFb1d',varargin{:});
            else
                error('SaivDr: Invalid arguments');
            end
        end

    end

end
