%CREATESAIVDRBLKS Script for Creating SaivDr Blockset 
%
% http://www.mathworks.co.jp/jp/help/matlab/matlab-unit-test-framework.html
%
% SVN identifier:
% $Id: createsaivdrblks.m 683 2015-05-29 08:22:13Z sho $
%
% Requirements: MATLAB R2014a
%
% Copyright (c) 2014-2015, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%

%% Close SaivDr Blockset Library if it's in memory
if bdIsLoaded('SaivDrBlockSet')
    close_system('SaivDrBlockSet')
end

%% Delete previous SaivDr Blockset Library if it exists
if exist('SaivDrBlockSet','file') == 4
    movefile('SaivDrBlockSet.slx','SaivDrBlockSet_bak.slx')
end

%% Prepare for Library 
h = new_system('SaivDrBlockSet','Library');

%% Location
winLeft = 60;
winTop  = 60;
nRows = 2;
nCols = 3;
moduleSep    = 40;
topMargin    = moduleSep;
leftMargin   = moduleSep;
moduleWidth  = 80;
moduleHeight = 60; 

%% Registoration of Modules
idx = 0;
% Registore 2-D DCT
idx = idx + 1;
iRow = floor((idx-1)/nCols)+1;
iCol = mod((idx-1),nCols)+1;
posL = leftMargin + (iCol-1)*(moduleWidth+moduleSep);
posT = topMargin  + (iRow-1)*(moduleHeight+moduleSep);
posR = posL + moduleWidth;
posB = posT + moduleHeight;
position = [ posL posT posR posB ];
add_block('Simulink/User-Defined Functions/MATLAB System',...
    'SaivDrBlockSet/DCT_2D','System',...
    'saivdr.embedded.ModuleBlockDct2d')
set_param('SaivDrBlockSet/DCT_2D',...
    'Position',position);

% Registore 2-D IDCT
idx = idx + 1;
iRow = floor((idx-1)/nCols)+1;
iCol = mod((idx-1),nCols)+1;
posL = leftMargin + (iCol-1)*(moduleWidth+moduleSep);
posT = topMargin  + (iRow-1)*(moduleHeight+moduleSep);
posR = posL + moduleWidth;
posB = posT + moduleHeight;
position = [ posL posT posR posB ];
add_block('Simulink/User-Defined Functions/MATLAB System',...
    'SaivDrBlockSet/IDCT_2D','System',...
    'saivdr.embedded.ModuleBlockIdct2d')
set_param('SaivDrBlockSet/IDCT_2D',...
    'Position',position);

% Registore Butterfly
idx = idx + 1;
iRow = floor((idx-1)/nCols)+1;
iCol = mod((idx-1),nCols)+1;
posL = leftMargin + (iCol-1)*(moduleWidth+moduleSep);
posT = topMargin  + (iRow-1)*(moduleHeight+moduleSep);
posR = posL + moduleWidth;
posB = posT + moduleHeight;
position = [ posL posT posR posB ];
add_block('Simulink/User-Defined Functions/MATLAB System',...
    'SaivDrBlockSet/Butterfly','System',...
    'saivdr.embedded.ModuleButterfly')
set_param('SaivDrBlockSet/Butterfly',...
    'Position',position);

% Registore Rotations
idx = idx + 1;
iRow = floor((idx-1)/nCols)+1;
iCol = mod((idx-1),nCols)+1;
posL = leftMargin + (iCol-1)*(moduleWidth+moduleSep);
posT = topMargin  + (iRow-1)*(moduleHeight+moduleSep);
posR = posL + moduleWidth;
posB = posT + moduleHeight;
position = [ posL posT posR posB ];
add_block('Simulink/User-Defined Functions/MATLAB System',...
    'SaivDrBlockSet/Rotations','System',...
    'saivdr.embedded.ModuleRotations')
set_param('SaivDrBlockSet/Rotations',...
    'Position',position);

% Registore Partial Delay
idx = idx + 1;
iRow = floor((idx-1)/nCols)+1;
iCol = mod((idx-1),nCols)+1;
posL = leftMargin + (iCol-1)*(moduleWidth+moduleSep);
posT = topMargin  + (iRow-1)*(moduleHeight+moduleSep);
posR = posL + moduleWidth;
posB = posT + moduleHeight;
position = [ posL posT posR posB ];
add_block('Simulink/User-Defined Functions/MATLAB System',...
    'SaivDrBlockSet/Partial_Delay','System',...
    'saivdr.embedded.ModulePartialDelay')
set_param('SaivDrBlockSet/Partial_Delay',...
    'Position',position);

% Registore Partial Line Buffer
idx = idx + 1;
iRow = floor((idx-1)/nCols)+1;
iCol = mod((idx-1),nCols)+1;
posL = leftMargin + (iCol-1)*(moduleWidth+moduleSep);
posT = topMargin  + (iRow-1)*(moduleHeight+moduleSep);
posR = posL + moduleWidth;
posB = posT + moduleHeight;
position = [ posL posT posR posB ];
add_block('Simulink/User-Defined Functions/MATLAB System',...
    'SaivDrBlockSet/Partial_Line_Buffer','System',...
    'saivdr.embedded.ModulePartialLineBuffer')
set_param('SaivDrBlockSet/Partial_Line_Buffer',...
    'Position',position);

%% Window Location
locL = winLeft;
locT = winTop;
locR = locL + nCols*(moduleWidth+moduleSep) +moduleSep + 60;
locB = locT + nRows*(moduleHeight+moduleSep)+moduleSep + 180;
set_param(h,'Location',[locL locT locR locB]); 

%% Save and oper SaivDr Blockset Library
save_system('SaivDrBlockSet')
open_system('SaivDrBlockSet')
