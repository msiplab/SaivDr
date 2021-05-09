function [fcnhandler,flag] = fcn_build_orthmtxgen_diff()
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
mexname = sprintf('%s_mex',bsfname);

if license('checkout','matlab_coder') % Coder is available
    cdir = pwd;
    saivdr_root = getenv('SAIVDR_ROOT');
    cd(saivdr_root)
    packagedir = './+saivdr/+dcnn/+mexsrcs';
    fbsfile = exist([packagedir '/' bsfname '.m'],'file');
    
    if fbsfile == 2
        
        outputdir = fullfile(saivdr_root,'mexcodes');
        %
        aAngles   = coder.typeof(single(0),[inf 1],[1 0]); %#ok
        aMus      = coder.typeof(single(0),[inf 1],[1 0]); %#ok
        aPdAng    = coder.typeof(uint32(0),1,0); %#ok
        aMtxPst   = coder.typeof(single(0),[inf inf],[1 1]); %#ok
        aMtxPre   = coder.typeof(single(0),[inf inf],[1 1]); %#ok
        % build mex
        if license('checkout','gpu_coder')  
            disp('GPU Coder')
            cfg = coder.gpuConfig('mex');   
        else
            cfg = coder.config('mex');
        end
        cfg.DynamicMemoryAllocation = 'AllVariableSizeArrays';%'Threshold';%'Off';
        cfg.GenerateReport = true;
        args = '{ aAngles, aMus, aPdAng, aMtxPst, aMtxPre }';
        seval = [ 'codegen -config cfg ' ' -o ' outputdir '/' mexname ' ' ...
            packagedir '/' bsfname '.m -args ' args];
        
        disp(seval)
        eval(seval)
        
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
