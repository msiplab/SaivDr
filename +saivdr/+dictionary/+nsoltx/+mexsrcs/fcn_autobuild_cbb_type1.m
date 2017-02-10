function [fcnhandler,flag] = fcn_autobuild_bb_type1(hChs)
%FCN_AUTOBUILD_BB_TYPE1
%
% SVN identifier:
% $Id: fcn_autobuild_bb_type1.m 683 2015-05-29 08:22:13Z sho $
%
% Requirements: MATLAB R2013b
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
% LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
%

% global isAutoBuildBbType1Locked
% if isempty(isAutoBuildBbType1Locked)
%     isAutoBuildBbType1Locked = false;
% end

bsfname = 'fcn_Order1BuildingBlockTypeI';
mexname = sprintf('%s_%d_%d_mex',bsfname,hChs,hChs);

ftypemex = exist(mexname, 'file');
if exist('getCurrentTask','file') == 2
    task = getCurrentTask();
else
    task = [];
end
if isempty(task) || task.ID == 1
% if ~isAutoBuildBbType1Locked
%     isAutoBuildBbType1Locked = true;
    if ftypemex ~= 3  && ... % MEX file doesn't exist
            license('checkout','matlab_coder') % Coder is available
        cdir = pwd;
        saivdr_root = getenv('SAIVDR_ROOT');
        cd(saivdr_root)
        packagedir = './+saivdr/+dictionary/+nsoltx/+mexsrcs';
        fbsfile = exist([packagedir '/' bsfname '.m'],'file');

        if fbsfile == 2

            outputdir = fullfile(saivdr_root,'mexcodes');
            %
            maxNCfs = 518400;
            %
            arrayCoefs = coder.typeof(complex(0),[2*hChs maxNCfs],[0 1]); %#ok
            paramMtxW = coder.typeof(double(0),hChs*[1 1],[0 0]); %#ok
            paramMtxU = coder.typeof(double(0),hChs*[1 1],[0 0]); %#ok
            paramAngles = coder.typeof(double(0), [floor(hChs/2),1],[0 0]); %#ok
            constHChs = coder.Constant(hChs); %#ok
            nshift = coder.typeof(int32(0),[1 1],[0 0]); %#ok
            % build mex
            cfg = coder.config('mex');
            cfg.DynamicMemoryAllocation = 'Threshold';
            cfg.GenerateReport = true;
            args = '{ arrayCoefs, paramMtxW, paramMtxU, paramAngles, constHChs, nshift }';
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
    %isAutoBuildBbType1Locked = false;
end

if ftypemex == 3 % MEX file exists

    fcnhandler = str2func(mexname);
    flag       = true;

else

    fcnhandler = [];
    flag       = false;

end
