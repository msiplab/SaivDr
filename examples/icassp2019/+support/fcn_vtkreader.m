function varargout = fcn_vtkreader(filename)
%FCN_VTKREADER VTK Reader for River CPS Project
%   
% Each file is in VTK format, and contain four physical quantities:
%
% - Water surface height
% - Water surface variation from the beginning
% - Water bed height
% - Water bed variation from the beginning,
%

%% File name
if nargin < 1
    filename = './DATA(20170906)/VTK/0100_trm.vtk';
end

%% File open
fileId = fopen(filename,'r');

%% Read header (skip)
for idx = 1:4
    fgetl(fileId);
end

%% Read DIMENSIONS 
line = fgetl(fileId);
str = extractAfter(line,'DIMENSIONS');
elements = strsplit(strtrim(str));
nElements = length(elements);
nDims = zeros(1,nElements);
for idx = 1:nElements
    nDims(idx) = str2double(elements{idx});
end

%% Read POINTS
line = fgetl(fileId);
str = extractAfter(line,'POINTS');
elements = strsplit(strtrim(str));
nPoints = str2double(elements{1});
%type = elements{2}

%% Read grids
x = zeros(nPoints,1);
y = zeros(nPoints,1);
%z = zeros(nPoints,1);
for iPoint = 1:nPoints
    line = fgetl(fileId);
    elements = strsplit(strtrim(line));
    x(iPoint) = str2double(elements{1});
    y(iPoint) = str2double(elements{2});
    %z(iPoint) = str2double(elements{3});
end

%% Read water surface height
dataFlag = false;
while(~dataFlag)
    line = fgetl(fileId);
    dataFlag = contains(line,'POINT_DATA');
end
str = extractAfter(line,'POINT_DATA');
elements = strsplit(strtrim(str));
nPoints = str2double(elements{1});

%% Read number of fields
line = fgetl(fileId);
str = extractAfter(line,'FIELD');
elements = strsplit(strtrim(str));
nFields = str2double(elements{2});

%%
fieldNames = cell(nFields,1);
fieldData  = cell(nFields,1);
for iField = 1:nFields
    line = strtrim(fgetl(fileId));
    elements = strsplit(line);    
    fieldNames{iField} = elements{1};
    nDims = [str2double(elements{2}) str2double(elements{3})];
    fieldData{iField} = zeros(nDims(1),nDims(2));
    for iPoint = 1:nPoints
        elements = strtrim(fgetl(fileId));
        fieldData{iField}(iPoint) = str2double(elements);
    end
end

%% File close
fclose(fileId);

%%
if nargout < 3
    varargout{1} = fieldData;
    varargout{2} = fieldNames;
else
    varargout{1} = x;
    varargout{2} = y;
    varargout{3} = fieldData;
    varargout{4} = fieldNames;
end

end