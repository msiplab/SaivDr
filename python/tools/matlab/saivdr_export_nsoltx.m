function saivdr_export_nsoltx(lppufb,file,varargin)
%SAIVDR_EXPORT_NSOLTX Export a 2-D Type-I nsoltx filter bank for saivdr.dcnn
%
%   saivdr_export_nsoltx(lppufb,file,'NumberOfLevels',nLevels) converts the
%   Angles and Mus of LPPUFB, a saivdr.dictionary.nsoltx 2-D Type-I
%   OvsdLpPuFb system (saivdr.dictionary.nsolt objects are upgraded), into
%   the parameters of the rotation layers of the network built by
%   fcn_creatensoltlgraph2d (or saivdr.dcnn.create_nsolt_2d in Python), and
%   saves them to the MAT-file FILE, which
%   saivdr.dcnn.matlab_compat.load_nsoltx reads in Python.
%
%   The columns of Angles and Mus correspond to W0, U0, Uh1, ..., UhNh,
%   Uv1, ..., UvNv, which are mapped to the layers V0 (W0 and U0), Vh1, ...,
%   VhNh, Vv1, ..., VvNv and their synthesis counterparts (~) of every level.
%   The resulting atoms coincide with those of LPPUFB up to translations by
%   multiples of the decimation factor.
%
% Copyright (c) 2026, Shogo MURAMATSU
p = inputParser;
addParameter(p,'NumberOfLevels',1)
parse(p,varargin{:})
nLevels = p.Results.NumberOfLevels;
if isa(lppufb,'saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dSystem')
    lppufb = saivdr.dictionary.utility.fcn_upgrade(lppufb);
end
nChs = get(lppufb,'NumberOfChannels');
if nChs(1) ~= nChs(2)
    error('Only Type-I NSOLT is supported')
end
ord = get(lppufb,'PolyPhaseOrder');
if any(mod(ord,2))
    error('Only even polyphase orders are supported')
end
dec = get(lppufb,'DecimationFactor');
angles = double(lppufb.Angles);
mus = double(lppufb.Mus);
noDcLeakage = isa(lppufb,'saivdr.dictionary.nsoltx.OvsdLpPuFb2dTypeIVm1System');
% (name, columns) of the rotation layers in each level
specs = {'V0',[1 2]};
for iOrd = 1:ord(2)
    specs(end+1,:) = {sprintf('Vh%d',iOrd),2+iOrd}; %#ok<AGROW>
end
for iOrd = 1:ord(1)
    specs(end+1,:) = {sprintf('Vv%d',iOrd),2+ord(2)+iOrd}; %#ok<AGROW>
end
n = 2*nLevels*size(specs,1);
s.names = cell(n,1);
s.angles = cell(n,1);
s.mus = cell(n,1);
s.nodcleakage = nan(n,1);
idx = 0;
for iLv = 1:nLevels
    for iSpec = 1:size(specs,1)
        cols = specs{iSpec,2};
        for suffix = {'','~'}
            idx = idx + 1;
            s.names{idx} = sprintf('Lv%d_Cmp1_%s%s',iLv,specs{iSpec,1},suffix{1});
            s.angles{idx} = reshape(angles(:,cols),[],1);
            s.mus{idx} = reshape(mus(:,cols),[],1);
            if iSpec == 1
                s.nodcleakage(idx) = double(noDcLeakage);
            end
        end
    end
end
s.number_of_channels = nChs;
s.decimation_factor = dec;
s.polyphase_order = ord;
s.number_of_levels = nLevels;
s.number_of_vanishing_moments = double(noDcLeakage);
save(file,'-struct','s','-v7');
end
