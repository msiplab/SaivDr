function [fcnhandler,flag] = fcn_autobuild(bsfname,ps,pa)
%FCN_AUTOBUILD
%
% SVN identifier:
% $Id: fcn_autobuild.m 683 2015-05-29 08:22:13Z sho $
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014-2015, Shogo MURAMATSU
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

% global isAutoBuildLocked
% if isempty(isAutoBuildLocked)
%     isAutoBuildLocked = false;
% end

mexname = sprintf('%s_%d_%d_mex',bsfname,ps,pa);
ftypemex = exist(mexname, 'file');
if exist('getCurrentTask','file') == 2
    task = getCurrentTask();
else
    task = [];
end
if isempty(task) || task.ID == 1
% if ~isAutoBuildLocked
%     isAutoBuildLocked = true;
    if ftypemex ~= 3  && ... % MEX file doesn't exist
            license('checkout','matlab_coder') % Coder is available
        
        cdir = pwd;
        saivdr_root = getenv('SAIVDR_ROOT');
        cd(saivdr_root)
        packagedir = './+saivdr/+dictionary/+nsolt/+mexsrcs';
        fbsfile = exist([packagedir '/' bsfname '.m'],'file');
        
        if fbsfile == 2
            
            outputdir = fullfile(saivdr_root,'mexcodes');
            %
            maxNCfs = 518400;
            %
            arrayCoefs = coder.typeof(double(0),[(ps+pa) maxNCfs],[0 1]); %#ok
            nRows = coder.typeof(int32(0),[1 1],[0 0]); %#ok
            nCols = coder.typeof(int32(0),[1 1],[0 0]); %#ok
            ord = coder.typeof(int32(0),[1 1],[0 0]); %#ok
            isPeriodicExt = coder.typeof(false,[1 1],[0 0]); %#ok
            paramMtx1 = coder.typeof(double(0),ps*[1 1],[0 0]); %#ok
            paramMtx2 = coder.typeof(double(0),pa*[1 1],[0 0]); %#ok
            % build mex
            cfg = coder.config('mex');
            cfg.DynamicMemoryAllocation = 'Off';
            cfg.GenerateReport = true;
            args = '{ arrayCoefs, nRows, nCols, paramMtx1, paramMtx2, isPeriodicExt }';
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
    %isAutoBuildLocked = false;
end

if ftypemex == 3 % MEX file exists
    
    fcnhandler = str2func(mexname);
    flag       = true;
    
else 

    fcnhandler = [];
    flag       = false;
end
