% DISP_SPECIFICATIONS Display design specifications
%
% This script displays some design specifications for DirLOT.
% This example is used for the script MIAN_PARDIRLOTDSGN.
%
% SVN identifier:
% $Id: disp_specifications.m 683 2015-05-29 08:22:13Z sho $
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
close all

%% Initial setting
transition = 0.25;   % Transition band width
nPoints = [128 128]; % Number of frequency sampling points

%% First case
import saivdr.dictionary.utility.Direction
dec = [2 2]; % Decimation factor
alpha = -2.0; % Deformation parameter
direction = Direction.HORIZONTAL; % Deformation direciton

import saivdr.dictionary.nsgenlotx.design.SubbandSpecification
sbsp = SubbandSpecification(...
    'Alpha',alpha,...
    'Direction',direction,...
    'DecimationFactor',dec,...
    'OutputMode','PassStopAssignMent'); % Subband specification
set(sbsp,'Transition',transition); % Set treansition band width

figure(1)
for idx = 1:prod(dec)
    spec = step(sbsp,nPoints,idx);
    subplot(dec(1),dec(2),idx)
    imshow(flipud(spec+1.0)/2.0);
end

%% Second case
dec = [4 4]; % Decimation factor
alpha = -2.0; % Deformation parameter
direction = Direction.HORIZONTAL; % Deformation direciton

sbsp = SubbandSpecification(...
    'Alpha',alpha,...
    'Direction',direction,...
    'DecimationFactor',dec,...
    'OutputMode','PassStopAssignment'); % Subband specification
set(sbsp,'Transition',transition); % Set treansition band width

figure(2)
for idx = 1:prod(dec)
    spec = step(sbsp,nPoints,idx);
    subplot(dec(1),dec(2),idx)
    imshow(flipud(spec+1.0)/2.0);
end
