%MAIN_VIEWPCD Script for monitoring PCD
%

%% Source & destination folder
srcfolder = RiverCpsConfig.SrcFolder;
dstfolder = './';
isAprx = false;
nummode = 6;
%
ts = 100;
tint = 10;

%
if isAprx
    avifile   = [dstfolder 'river_aprx_r' num2str(nummode,'%02d')];
    te = 430;
else
    avifile   = [dstfolder 'river' ];
    te = 440;    
end

%%
if isempty(avifile)
    isWriteVideo = false;
else
    isWriteVideo = true;
end

if isWriteVideo
    vwObj = VideoWriter(avifile,'MPEG-4');
    vwObj.FrameRate = 5;
    vwObj.open();
end

%% Field
fieldList = {
    'water_level' ...
    'bed_level' ...
    ...'depth_of_water' ...
    ...'deviation_of_bed_level'
    };
nFields = length(fieldList);

%%
player = pcplayer([0 10],[-.5 .5],[0 0.08]);
player.Axes.DataAspectRatio = [20 20 1];
player.Axes.XDir = 'reverse';

%%
for t = ts:tint:te
    if isAprx
        filename = sprintf('%04d_aprx_r%02d',t, nummode);
    else
        filename = sprintf('%04d_trm',t);
    end
    disp(filename)
    field = fieldList{1};
    ptCloud = pcread([ srcfolder field '_' filename '.pcd' ]);
    for iField = 2:nFields
        field = fieldList{iField};
        ptCloudAdd = pcread([ srcfolder field '_' filename '.pcd' ]);
        ptCloud = pcmerge(ptCloud,ptCloudAdd,1e-3);
    end
    %
    %if isOpen(player)
        view(player,ptCloud);
    %end
    %
    if isWriteVideo
        frame = getframe(player.Axes);
        vwObj.writeVideo(frame)
    end
end

%%
if isWriteVideo
    vwObj.close();
end