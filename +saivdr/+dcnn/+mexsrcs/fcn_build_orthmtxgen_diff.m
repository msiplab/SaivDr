function [fcnhandler,flag] = fcn_build_orthmtxgen_diff(datatype,useGpuArray)
%FCN_BUILD_ORTHMTXGEN_DIFF
%
% Requirements: MATLAB R2021a
%
% Copyright (c) 2021, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
bsfname = 'fcn_orthmtxgen_diff';

if nargin < 1 || isempty(datatype)
    datatype = 'double';
end
if nargin < 2 || useGpuArray
    device = 'gpu';
else
    device = 'cpu';
end

mexname = sprintf('%s_%s_on_%s_mex',bsfname, datatype, device);

if license('checkout','matlab_coder') % Coder is available
    cdir = pwd;
    saivdr_root = getenv('SAIVDR_ROOT');
    cd(saivdr_root)
    packagedir = './+saivdr/+dcnn/+mexsrcs';
    fbsfile = exist([packagedir '/' bsfname '.m'],'file');
    
    if fbsfile == 2
        
        outputdir = fullfile(saivdr_root,'mexcodes');
        %
        % build mex
        codegenskip = false;
        if license('checkout','gpu_coder')
            disp('GPU Coder')
            cfg = coder.gpuConfig('mex');
        elseif strcmp(device,'cpu')
            cfg = coder.config('mex');
        else
            codegenskip = true;
        end
        if strcmp(device,'cpu') % on CPU
            aAngles   = coder.typeof(single(0),[inf 1],[1 0]); %#ok
            aMus      = coder.typeof(single(0),[inf 1],[1 0]); %#ok
            aPdAng    = coder.typeof(uint32(0),1,0); %#ok
            aMtxPst   = coder.typeof(single(0),[inf inf],[1 1]); %#ok
            aMtxPre   = coder.typeof(single(0),[inf inf],[1 1]); %#ok
            cfg.DynamicMemoryAllocation = 'AllVariableSizeArrays';%'Threshold';%'Off';
        else % on GPU
            nChs = 64;
            maxAngs = (nChs-2)*nChs/8;
            maxMus = nChs/2;
            aAngles   = coder.typeof(gpuArray(single(0)),[maxAngs 1],[1 0]); %#ok
            aMus      = coder.typeof(gpuArray(single(0)),[maxMus 1],[1 0]); %#ok
            aPdAng    = coder.typeof(uint32(0),1,0); %#ok
            aMtxPst   = coder.typeof(gpuArray(single(0)),[maxMus maxMus],[1 1]); %#ok
            aMtxPre   = coder.typeof(gpuArray(single(0)),[maxMus maxMus],[1 1]); %#ok
            cfg.DynamicMemoryAllocation = 'Off';
        end
        
        if codegenskip
            disp('Skipping code generation')
        else
            cfg.GenerateReport = true;
            args = '{ aAngles, aMus, aPdAng, aMtxPst, aMtxPre }';
            seval = [ 'codegen -config cfg ' ' -o ' outputdir '/' mexname ' ' ...
                packagedir '/' bsfname '.m -args ' args];
            
            disp(seval)
            eval(seval)
        end
    else
        error('SaivDr: Invalid argument')
    end
    
    cd(cdir)
end
ftypemex = exist(mexname, 'file');

if ftypemex == 3 % MEX file exists
    
    fcnhandler = str2func(mexname);
    flag       = true;
    
else
    
    fcnhandler = [];
    flag       = false;
    
end

