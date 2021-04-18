function [fcnhandler,flag] = fcn_build_gradevalsteps2d()
%%FCN_BUILD_GRADEVALSGTEPS2D
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2015-2021, Shogo MURAMATSU
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

bsfname = 'fcn_GradEvalSteps2dCodeGen';
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
        aCoefsB   = coder.typeof(double(0),[inf inf],[1 1]); %#ok
        aCoefsC   = coder.typeof(double(0),[inf inf],[1 1]); %#ok
        aScale    = coder.typeof(uint32(0),[1 2],[0 0]); %#ok
        aPmCoefs  = coder.typeof(double(0),[inf 1],[1 0]); %#ok
        aAngs     = coder.typeof(double(0),[inf inf],[1 1]); %#ok
        aMus      = coder.typeof(double(0),[inf inf],[1 1]); %#ok
        aNch      = coder.typeof(double(0),[1 2],[0 0]); %#ok
        aOrd      = coder.typeof(uint32(0),[1 2],[0 0]); %#ok
        aFpe      = coder.typeof(logical(0),1,0); %#ok
        aNdc      = coder.typeof(logical(0),1,0); %#ok
        % build mex
        cfg = coder.config('mex');
        cfg.DynamicMemoryAllocation = 'AllVariableSizeArrays';%'Threshold';%'Off';
        cfg.GenerateReport = true;
        args = '{ aCoefsB, aCoefsC, aScale, aPmCoefs, aAngs, aMus, aNch, aOrd, aFpe, aNdc }';
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
