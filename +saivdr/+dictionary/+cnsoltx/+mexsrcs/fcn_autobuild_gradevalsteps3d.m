function [fcnhandler,flag] = fcn_autobuild_gradevalsteps3d(nch,ord)
%FCN_AUTOBUILD_GRADEVALSGTEPS3D
%
% SVN identifier:
% $Id: fcn_autobuild_gradevalsteps3d.m 866 2015-11-24 04:29:42Z sho $
%
% Requirements: MATLAB R2013b
%
% Copyright (c) 2015, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
%

%global isAutoBuildAtomCnc3dLocked
%if isempty(isAutoBuildAtomCnc3dLocked)
%    isAutoBuildAtomCnc3dLocked = false;
%end

bsfname = 'fcn_GradEvalSteps3d';
mexname = sprintf('%s_cs%da%d_ov%dh%dd%d_mex',bsfname,...
    nch(1),nch(2),ord(1),ord(2),ord(3));

ftypemex = exist(mexname, 'file');
if exist('getCurrentTask','file') == 2
    task = getCurrentTask();
else
    task = [];
end
if isempty(task) || task.ID == 1
%if ~isAutoBuildAtomCnc3dLocked
%    isAutoBuildAtomCnc3dLocked = true;
    if ftypemex ~= 3  && ... % MEX file doesn't exist
            license('checkout','matlab_coder') % Coder is available
        cdir = pwd;
        saivdr_root = getenv('SAIVDR_ROOT');
        cd(saivdr_root)
        packagedir = './+saivdr/+dictionary/+cnsoltx/+mexsrcs';
        fbsfile = exist([packagedir '/' bsfname '.m'],'file');
        
        if fbsfile == 2
            
            outputdir = fullfile(saivdr_root,'mexcodes');
            %
            %maxNCfs = 518400;
            %nPmCoefs = (nch(1)^2 +nch(2)^2)*(sum(ord)/2+1);
            %
            ps = nch(1);
            %pa = nch(2);
            nAngsPm   = ps*(ps-1)/2;
            nMusPm    = ps;            
            %
            aCoefsB   = coder.typeof(double(0),[sum(nch) Inf],[0 1]); %#ok
            aCoefsC   = coder.typeof(double(0),[sum(nch) Inf],[0 1]); %#ok
            aScale    = coder.typeof(uint32(0),[1 3],[0 0]); %#ok
            aPmCoefs  = coder.typeof(double(0),[Inf 1],[1 0]); %#ok 
            aAngs     = coder.typeof(double(0),[nAngsPm (sum(ord)+2)],[0 0]); %#ok                        
            aMus      = coder.typeof(double(0),[nMusPm  (sum(ord)+2)],[0 0]); %#ok                                    
            cNch      = coder.Constant(nch); %#ok            
            cOrd      = coder.Constant(ord); %#ok
            aFpe      = coder.typeof(logical(0),[1 1],[0 0]); %#ok
            aNdc      = coder.typeof(logical(0),[1 1],[0 0]); %#ok
            % build mex
            cfg = coder.config('mex');
            cfg.DynamicMemoryAllocation = 'AllVariableSizeArrays';%'Threshold';%'Off';
            cfg.GenerateReport = true;
            args = '{ aCoefsB, aCoefsC, aScale, aPmCoefs, aAngs, aMus, cNch, cOrd, aFpe, aNdc }';
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
    %isAutoBuildAtomCnc3dLocked = false;
end

if ftypemex == 3 % MEX file exists
    
    fcnhandler = str2func(mexname);
    flag       = true;
    
else 

    fcnhandler = [];
    flag       = false;

end
