classdef NsGenLotFactory
    %NSGENLOTFACTORY  Factory class of NS-GenLOTs
    %
    % Requirements: MATLAB R2015b
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
    % http://msiplab.eng.niigata-u.ac.jp/                  Niigata, 950-2181, JAPAN
    %
    
    methods (Static = true)
        
        function value = createLpPuFb2dSystem(varargin)
            import saivdr.dictionary.nsoltx.*
            import saivdr.dictionary.nsgenlotx.*
            if nargin < 1
                value = OvsdLpPuFb2dTypeIVm1System();
            elseif isa(varargin{1},'saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem')
                value = clone(varargin{1});
            else 
                p = inputParser;
                addOptional(p,'DecimationFactor',[2 2]);
                addOptional(p,'PolyPhaseOrder',[]);
                addOptional(p,'NumberOfVanishingMoments',1);
                addOptional(p,'TvmAngleInDegree',[]);
                addOptional(p,'OutputMode',[]);
                parse(p,varargin{:});
                vm = p.Results.NumberOfVanishingMoments;
                angleTvm = p.Results.TvmAngleInDegree;
                % 
                newidx = 1;
                oldidx = 1;
                while oldidx < nargin+1
                    if ischar(varargin{oldidx}) && ...
                           ( strcmp(varargin{oldidx},'NumberOfVanishingMoments') )
                        oldidx = oldidx + 1;
                    elseif strcmp(varargin{oldidx},'TvmAngleInDegree') && ...
                            isempty(angleTvm)
                        oldidx = oldidx + 1;                        
                    else
                        argin{newidx} = varargin{oldidx};
                        newidx = newidx + 1;
                    end
                    oldidx = oldidx + 1;
                end
                if vm == 0
                    if exist('argin','var') ~= 1
                        value = OvsdLpPuFb2dTypeIVm0System();
                    else
                        value = OvsdLpPuFb2dTypeIVm0System(argin{:});
                    end
                elseif vm == 1
                    if exist('argin','var') ~= 1
                        value = OvsdLpPuFb2dTypeIVm1System();
                    else
                        value = OvsdLpPuFb2dTypeIVm1System(argin{:});
                    end
                elseif vm == 2
                    if isempty(angleTvm)
                        if exist('argin','var') ~= 1
                            value = LpPuFb2dVm2System();
                        else
                            value = LpPuFb2dVm2System(argin{:});
                        end
                    elseif isscalar(angleTvm)
                        value = LpPuFb2dTvmSystem(argin{:});
                    end
                else
                    id = 'SaivDr:IllegalArgumentException';
                    msg = 'Unsupported type of vanishing moments';
                    me = MException(id, msg);
                    throw(me);
                end
            end
        end
        
    end
    
end

