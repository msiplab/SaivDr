function [fcnhandler,flag] = fcn_build_atomcnc3d()
%FCN_AUTOBUILD_ATOMCNC3D
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014-2023, Shogo MURAMATSU
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
isOlderThanR2023b = isMATLABReleaseOlderThan("R2023b"); % 'DynamicMemoryAllocation' -> 'EnableDynamicMemoryAllocation' と 'DynamicMemoryAllocationThreshold'

bsfname = 'fcn_NsoltAtomConcatenator3dCodeGen';
mexname = sprintf('%s_mex',bsfname);

if license('checkout','matlab_coder') % Coder is available
    cdir = pwd;
    saivdr_root = getenv('SAIVDR_ROOT');
    cd(saivdr_root)
    packagedir = './+saivdr/+dictionary/+nsoltx/+mexsrcs';
    fbsfile = exist([packagedir '/' bsfname '.m'],'file');
    
    if fbsfile == 2
        
        outputdir = fullfile(saivdr_root,'mexcodes');
        %
        aCoefs   = coder.typeof(double(0),[inf inf],[1 1]); %#ok
        aScale   = coder.typeof(uint32(0),[1 3],[0 0]); %#ok
        aPmCoefs = coder.typeof(double(0),[Inf 1],[1 0]); %#ok
        aNch     = coder.typeof(double(0),[1 2],[0 0]); %#ok
        aOrd     = coder.typeof(uint32(0),[1 3],[0 0]); %#ok
        aFpe     = coder.typeof(logical(0),1,0); %#ok
        % build mex
        cfg = coder.config('mex');
        if isOlderThanR2023b
            cfg.DynamicMemoryAllocation = 'AllVariableSizeArrays';%'Threshold';%'Off';
        else
            cfg.EnableDynamicMemoryAllocation = true;
            cfg.DynamicMemoryAllocationThreshold = 2000000000; % in bytes
        end
        cfg.GenerateReport = true;
        args = '{ aCoefs, aScale, aPmCoefs, aNch, aOrd, aFpe }';
        seval = [ 'codegen -config cfg ' ' -o ''' outputdir '/' mexname ''' ' ...
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
