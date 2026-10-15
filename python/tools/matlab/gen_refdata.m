function gen_refdata(varargin)
%GEN_REFDATA Generate reference data for the PyTorch port of saivdr.dcnn
%
%   gen_refdata() writes MAT-files to python/tests/refdata, which the
%   Python tests (python/tests/test_refdata_*.py) compare with the PyTorch
%   implementation.
%
%   gen_refdata('Name',Value,...) accepts
%     'OutDir'  : output folder (default: <this folder>/../../tests/refdata)
%     'Groups'  : string array of groups to generate, from
%                 "layers", "networks2d", "networks3d", "atoms", "params",
%                 "nsoltx", "degradation", "ista"
%                 (default: all)
%     'Filter'  : regular expression on the case names (default: '.*')
%
%   Each case is generated with its own fixed random seed. Arrays are saved
%   in the MATLAB layout; the Python side converts them with
%   saivdr.dcnn.matlab_compat.from_matlab_array.
%
%   Run from the SaivDr root folder after setpath.
%
% Requirements: MATLAB R2022b or later, Deep Learning Toolbox
%
% Copyright (c) 2026, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/

p = inputParser;
here = fileparts(mfilename('fullpath'));
addParameter(p,'OutDir',fullfile(here,'..','..','tests','refdata'))
addParameter(p,'Groups',["layers","networks2d","networks3d","atoms","params",...
    "nsoltx","degradation","ista"])
addParameter(p,'Filter','.*')
parse(p,varargin{:})
outDir = char(p.Results.OutDir);
groups = string(p.Results.Groups);
filter = p.Results.Filter;
addpath(here)

manifest = struct('matlab',version,'saivdr_commit',gitcommit_(here),...
    'generated',char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ss')),...
    'cases',{{}});
if any(groups=="layers")
    manifest.cases = [manifest.cases; run_(refgen.layercases(),@refgen.genlayer,...
        fullfile(outDir,'layers'),filter)];
end
if any(groups=="networks2d")
    manifest.cases = [manifest.cases; run_(refgen.networkcases2d(),@refgen.gennetwork,...
        fullfile(outDir,'networks2d'),filter)];
end
if any(groups=="networks3d")
    manifest.cases = [manifest.cases; run_(refgen.networkcases3d(),@refgen.gennetwork,...
        fullfile(outDir,'networks3d'),filter)];
end
if any(groups=="atoms")
    manifest.cases = [manifest.cases; run_(refgen.atomcases(),@refgen.genatoms,...
        fullfile(outDir,'atoms'),filter)];
end
if any(groups=="params")
    manifest.cases = [manifest.cases; run_(refgen.paramcases(),@refgen.genparams,...
        fullfile(outDir,'params'),filter)];
end
for group = ["nsoltx","degradation","ista"]
    if any(groups==group)
        manifest.cases = [manifest.cases; run_(feval(['refgen.' char(group) 'cases']),...
            str2func(['refgen.gen' char(group)]),fullfile(outDir,char(group)),filter)];
    end
end
% Merge with an existing manifest so that partial regeneration keeps the others
manifestFile = fullfile(outDir,'manifest.json');
if isfile(manifestFile)
    old = jsondecode(fileread(manifestFile));
    if isfield(old,'cases') && ~isempty(old.cases)
        oldCases = old.cases;
        if isstruct(oldCases), oldCases = num2cell(oldCases); end
        newFiles = cellfun(@(c) c.file,manifest.cases,'UniformOutput',false);
        keep = cellfun(@(c) ~any(strcmp(c.file,newFiles)),oldCases);
        manifest.cases = [oldCases(keep); manifest.cases];
    end
end
files = cellfun(@(c) c.file,manifest.cases,'UniformOutput',false);
[~,order] = sort(files);
manifest.cases = manifest.cases(order);
fid = fopen(manifestFile,'w');
fprintf(fid,'%s\n',jsonencode(manifest,'PrettyPrint',true));
fclose(fid);
fprintf('Wrote %d cases to %s\n',numel(manifest.cases),outDir);
end

function entries = run_(cases,fcn,folder,filter)
if ~isfolder(folder)
    mkdir(folder);
end
entries = {};
for iCase = 1:numel(cases)
    c = cases{iCase};
    if isempty(regexp(c.name,filter,'once'))
        continue
    end
    rng(refgen.seed(c.name),'twister');
    s = fcn(c);
    s.meta_name = c.name;
    file = fullfile(folder,[c.name '.mat']);
    save(file,'-struct','s','-v7');
    [~,group] = fileparts(folder);
    entries{end+1,1} = struct('file',[group '/' c.name '.mat'],...
        'sha256',refgen.sha256(file)); %#ok<AGROW>
    fprintf('  %s\n',c.name);
end
end

function sha = gitcommit_(here)
[status,sha] = system(sprintf('git -C "%s" rev-parse HEAD',here));
if status ~= 0
    sha = 'unknown';
end
sha = strtrim(sha);
end
