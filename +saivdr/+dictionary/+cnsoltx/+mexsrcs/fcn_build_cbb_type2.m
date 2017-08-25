function [fcnhandler,flag] = fcn_build_cbb_type2()
%FCN_AUTOBUILD_BB_TYPE2
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014-2017, Shogo MURAMATSU
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

bsfname = 'fcn_Order2CplxBuildingBlockTypeII';
mexname = sprintf('%s_mex',bsfname);

if license('checkout','matlab_coder') % Coder is available
    cdir = pwd;
    saivdr_root = getenv('SAIVDR_ROOT');
    cd(saivdr_root)
    packagedir = './+saivdr/+dictionary/+cnsoltx/+mexsrcs';
    fbsfile = exist([packagedir '/' bsfname '.m'],'file');
    
    if fbsfile == 2
        
        outputdir = fullfile(saivdr_root,'mexcodes');
        %
        arrayCoefs = coder.typeof(complex(double(0)),[inf inf],[1 1]); %#ok
        paramMtxU1 = coder.typeof(double(0),[inf inf],[1 1]); %#ok
        paramMtxW1 = coder.typeof(double(0),[inf inf],[1 1]); %#ok
        paramAngB1 = coder.typeof(double(0),[inf,1],[0 0]); %#ok
        paramMtxU2 = coder.typeof(double(0),[inf inf],[1 1]); %#ok
        paramMtxW2 = coder.typeof(double(0),[inf inf],[1 1]); %#ok
        paramAngB2 = coder.typeof(double(0),[inf,1],[0 0]); %#ok
        aP         = coder.typeof(double(0),1,0); %#ok
        nshift     = coder.typeof(int32(0),1,0); %#ok
        % build mex
        cfg = coder.config('mex');
        cfg.DynamicMemoryAllocation = 'AllVariableSizeArrays';%'Threshold';
        cfg.GenerateReport = true;
        args = '{ arrayCoefs, paramMtxW1, paramMtxU1, paramAngB1, paramMtxW2, paramMtxU2, paramAngB2, aP, nshift }';
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
