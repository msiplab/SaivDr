%MAIN_VTK2PCD Script for converting VTK to PCD
%


%%
srcfolder = RiverCpsConfig.VtkFolder;
dstfolder = RiverCpsConfig.DstFolder;

%%
isVisible = true;
%
if isVisible
    xrange = [0 10];
    yrange = [-.5 .5];
    zrange = [0 0.08];
    player = pcplayer(xrange,yrange,zrange);
    player.Axes.DataAspectRatio = [20 20 1];
    player.Axes.XDir = 'reverse';
    %player.Axes.Colormap = flipud(parula);
    %player.Axes.CLim = zrange;
end

%% Base color for colorizatin
fieldcolor = containers.Map(...
    {'water_level', 'bed_level', 'depth_of_water', 'deviation_of_bed_level'},...
    {[0 128 255], [128 64 64], [0 128 192], [255 128 0]});

%%
for t = 100:10:440
    % Set file name for the first data
    filename = sprintf('%04d_trm',t);
    disp(filename)
    
    % Read VTK data
    [tmpx,tmpy,fieldData,fieldList] = ...
        support.fcn_vtkreader([srcfolder filename '.vtk']);
    
    % Grid alignment
    [~,locs] = findpeaks(xcorr(tmpx));
    distance = locs(2)-locs(1);
    width = numel(fieldData{1})/distance;
    x = reshape(tmpx,distance,width);
    y = reshape(tmpy,distance,width);
    
    % Convert VTK field data to z
    nFields = length(fieldData);
    for iField = 1:nFields
        z = fieldData{iField};
        z = reshape(z,distance,width);
        xyzPoints = cat(3,x,y,z);
        %xyzPoints(:,:,1) = x;
        %xyzPoints(:,:,2) = y;
        %xyzPoints(:,:,3) = z;
        ptCloud = pointCloud(xyzPoints);
        
        % Colorization
        %{
        frgb = fieldcolor(fieldList{iField})/255;
        fhsv = rgb2hsv(frgb);
        zh = fhsv(1)*ones(size(z));
        zs = fhsv(2)*ones(size(z));
        zv = fhsv(3)+imfilter(z/zrange(2),...
            [8 0 0; 0 0 0 ; 0 0 -8],'sym');
        ptCloud.Color = im2uint8(hsv2rgb(cat(3,zh,zs,zv)));
        %}
        basecolor = fieldcolor(fieldList{iField});
        ptCloud.Color = support.fcn_pointcolor(z,basecolor,zrange);
        
        % Output PCD
        pcwrite(ptCloud,...
            [ dstfolder fieldList{iField} '_' filename '.pcd' ])
    end
    
    % Visualization
    if isVisible && isOpen(player)
        field = fieldList{1};
        ptCloud = pcread(...
            [ dstfolder field '_' filename '.pcd' ]);
        for iField = 2:nFields
            field = fieldList{iField};
            ptCloudAdd = pcread(...
                  [ dstfolder fieldList{iField} '_' filename '.pcd' ]);
            ptCloud = pcmerge(ptCloud,ptCloudAdd,1e-3);
        end
        view(player,ptCloud);
    end
end
