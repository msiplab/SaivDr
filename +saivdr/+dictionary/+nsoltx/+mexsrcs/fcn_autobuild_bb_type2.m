function [fcnhandler,flag] = fcn_autobuild_bb_type2(fhChs)
%FCN_AUTOBUILD_BB_TYPE2
%
% SVN identifier:
% $Id: fcn_autobuild_bb_type2.m 683 2015-05-29 08:22:13Z sho $
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

% global isAutoBuildBbType2Locked
% if isempty(isAutoBuildBbType2Locked)
%     isAutoBuildBbType2Locked = false;
% end

bsfname = 'fcn_Order2BuildingBlockTypeII';
mexname = sprintf('%s_%d_%d_mex',bsfname,fhChs+1,fhChs);

ftypemex = exist(mexname, 'file');
if exist('getCurrentTask','file') == 2
    task = getCurrentTask();
else
    task = [];
end
if isempty(task) || task.ID == 1
% if ~isAutoBuildBbType2Locked
%     isAutoBuildBbType2Locked = true;
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
            arrayCoefs = coder.typeof(complex(0),[(2*fhChs+1) maxNCfs],[0 1]); %#ok
            paramMtxHW = coder.typeof(double(0),(fhChs+1)*[1 1],[0 0]);
            paramMtxHU = coder.typeof(double(0),(fhChs+1)*[1 1],[0 0]);
            paramAngles2 = coder.typeof(double(0),[floor(fhChs/2),1],[0 0]);
            paramMtxW = coder.typeof(double(0),fhChs*[1 1],[0 0]); %#ok
            paramMtxU = coder.typeof(double(0),fhChs*[1 1],[0 0]); %#ok
            paramAngles1 = coder.typeof(double(0),[floor(fhChs/2),1],[0 0]);
            constP = coder.Constant(fhChs); %#ok
            nshift = coder.typeof(int32(0),[1 1],[0 0]); %#ok
            % build mex
            cfg = coder.config('mex');
            cfg.DynamicMemoryAllocation = 'Threshold';
            cfg.GenerateReport = true;
            args = '{ arrayCoefs, paramMtxHW, paramMtxHU, paramAngles2, paramMtxW, paramMtxU, paramAngles1, constP, nshift }';
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
    %isAutoBuildBbType2Locked = false;
end

if ftypemex == 3 % MEX file exists

    fcnhandler = str2func(mexname);
    flag       = true;

else

    fcnhandler = [];
    flag       = false;

end
