% MAIN_PARDIRLOTDSGN DirLOT design process 
%
% This script executes design process of DirLOT with frequency 
% domain specification.
%
% Shogo Muramatsu, Dandan Han, Tomoya Kobayashi and Hisakazu Kikuchi: 
%  ''Directional Lapped Orthogonal Transform: Theory and Design,'' 
%  IEEE Trans. on Image Processing, Vol.21, No.5, pp.2434-2448, May 2012.
%  (DOI: 10.1109/TIP.2011.2182055)
%
% SVN identifier:
% $Id: main_pardirlotdsgn.m 683 2015-05-29 08:22:13Z sho $
%
% Requirements: MATLAB R2015b, Global optimization toolbox
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

%% Parameter settings
nPoints = [ 128 128 ];
kl  = [ 3 3 ];
phi = [ -30 0 30 60 90 120 ];

params.Display = 'final';
params.useParallel = 'always';
params.plotFcn = @gaplotbestf;
params.populationSize = 16;
params.eliteCount = 2;
params.generations = 40;
params.stallGenLimit = 20;
params.mutationFcn = @mutationgaussian;
params.nVm = 2;
params.dec = [ 2 2 ];
params.costCls = 'AmplitudeErrorEnergy';

%% Optimization
import saivdr.dictionary.nsgenlotx.design.SubbandSpecification
import saivdr.dictionary.utility.Direction
nReps = 4;
startOrd = 2; % Min. polyphase order
endOrd   = 4; % Max. polyphase order
stepOrd  = 2;
for iRep = 1:nReps
    for swmus = 0:1
        params.swmus = swmus;
        for ord = startOrd:stepOrd:endOrd
            params.ord = [ ord ord ];
            fprintf('dec=[%d %d], ord=[%d %d]\n',...
                params.dec(1),params.dec(2),...
                params.ord(1),params.ord(2));
            
            % VM2
            sbsp = SubbandSpecification(...
                'OutputMode','AmplitudeSpecification');
            for idx = 1:prod(params.dec)
                [as, sbIdx] = step(sbsp,nPoints,idx,kl);
                params.spec(:,:,sbIdx) = as;
            end
            params.phi = [];
            params.dt  = [ 1 1 ];
            support.fcn_updatedirlot(params);
            params.dt  = [ 1 -1 ];
            support.fcn_updatedirlot(params);
            params.dt  = [ -1 1 ];
            support.fcn_updatedirlot(params);
            params.dt  = [ -1 -1 ];
            support.fcn_updatedirlot(params);
            
            % TVM
            dec = params.dec(1);
            for iPhi = 1:length(phi)

                params_ = params;
                params_.phi =  mod(phi(iPhi)+45,180)-45;

                if params_.phi >= -45 && params_.phi < 45
                    if params_.phi == 0
                        alpha = 0;
                    else
                        alpha = cot(params_.phi*pi/180);
                    end
                    direction = Direction.HORIZONTAL;
                else
                    if params_.phi == 90
                        alpha = 0;
                    else
                        alpha = tan(params_.phi*pi/180);
                    end                    
                    direction = Direction.VERTICAL;
                end
                sbsp = SubbandSpecification(...
                    'Alpha',alpha,...
                    'Direction',direction,...
                    'OutputMode','AmplitudeSpecification');
                for idx = 1:prod(params_.dec)
                    [as, sbIdx] = step(sbsp,nPoints,idx,kl);
                    params_.spec(:,:,sbIdx) = as;
                end
                params_.dt = 1;
                support.fcn_updatedirlot(params_);
                params_.dt = -1;
                support.fcn_updatedirlot(params_);
            end
        end
    end
end
