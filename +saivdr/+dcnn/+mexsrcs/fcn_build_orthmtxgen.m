function [fcnhandler,flag] = fcn_build_orthmtxgen(datatype,useGpuArray)
%%FCN_BUILD_ORTHMTXGEN
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
bsfname = 'fcn_orthmtxgen';

if nargin < 1 || isempty(datatype)
    datatype = 'double';
end
if nargin < 2 || useGpuArray
    device = 'gpu';
else
    device = 'cpu';
end

% In R2021b, by default, code generation supports implicit expansion.
isLessThanR2021b = verLessThan('matlab','9.11'); % Effective only for CPU

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
            cfg = coder.gpuConfig('mex');
        elseif strcmp(device,'cpu')
            cfg = coder.config('mex');
        else
            codegenskip = true;
        end
        if strcmp(device,'cpu') % on CPU
            aAngles   = coder.typeof(cast(0,datatype),[inf 1],[1 0]); %#ok
            aMus      = coder.typeof(cast(0,datatype),[inf 1],[1 0]); %#ok
            cUseGpu   = coder.Constant(false); %#ok
            cIsLt21b  = coder.Constant(isLessThanR2021b); %#ok                        
            cfg.DynamicMemoryAllocation = 'AllVariableSizeArrays';%'Threshold';%'Off';
            cfg.ConstantInputs = 'Remove';
        elseif ~codegenskip % on GPU
            disp('GPU Coder')            
            nChs = 64;
            maxAngs = (nChs-2)*nChs/8;
            maxMus = nChs/2;
            aAngles   = coder.typeof(gpuArray(cast(0,datatype)),[maxAngs 1],[1 0]); %#ok
            aMus      = coder.typeof(gpuArray(cast(0,datatype)),[maxMus 1],[1 0]); %#ok
            cUseGpu   = coder.Constant(true); %#ok
            cIsLt21b  = coder.Constant(isLessThanR2021b); %#ok                        
            cfg.DynamicMemoryAllocation = 'Off';
            cfg.ConstantInputs = 'Remove';
        end
        
        if codegenskip
            disp('Skipping code generation')
        else
            cfg.GenerateReport = true;
            args = '{ aAngles, aMus, cUseGpu, cIsLt21b }';
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
