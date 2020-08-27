%MAIN_VIEWVTK Script for monitoring VTK
%

%% Source folder
srcfolder = RiverCpsConfig.VtkFolder;

%% Prepare point cloud player
player = pcplayer([0 10],[-.5 .5],[0 0.08]);
player.Axes.DataAspectRatio = [20 20 1];
player.Axes.XDir = 'reverse';
player.Axes.Colormap = flipud(parula);
player.Axes.CLim = [ 0 0.08 ];

%% Read sequencial VTK data
t = 100;
%
if isOpen(player)
    filename = sprintf('%04d_trm',t);
    disp(filename)
    [x,y,fieldData,fieldNames] = ...
        support.fcn_vtkreader([srcfolder filename '.vtk']);
    % Construct a point cloud object
    nFields = length(fieldData);
    xyzPoints = cell(nFields,1);
    for iField = 1:nFields
        z = fieldData{iField};
        xyzPoints{iField} = [ x(:) y(:) z(:) ];
    end
    ptCloud = pointCloud(cell2mat(xyzPoints));
    %
    view(player,ptCloud)
end

for t = 110:10:440
    %
    if isOpen(player)
        filename = sprintf('%04d_trm',t);
        disp(filename)
        [fieldData,fieldNames] = support.fcn_vtkreader([srcfolder filename '.vtk']);
        % Construct a point cloud object
        for iField = 1:nFields
            z = fieldData{iField};
            xyzPoints{iField} = [ x(:) y(:) z(:) ];
        end
        ptCloud = pointCloud(cell2mat(xyzPoints));
        %
        view(player,ptCloud)
    end
end