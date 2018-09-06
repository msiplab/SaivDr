classdef VolumetricDataVisualizer < matlab.System
    % VOLUMETRICDATAVISUALIZER
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.
    %
    % Inspired by VOL3D
    %
    %   https://jp.mathworks.com/matlabcentral/fileexchange/22940-vol3d-v2
    %
    
    % Public, tunable properties
    properties
        SlicePlane = 'XY'
        VRange     = [ 0 1 ]
        Scale      = 1
        ColorMap
        IsVerbose
        ImageObject
    end
    
    properties (Nontunable)
        Texture    = '2D'
        XData
        YData
        ZData
        Alpha
        Opts
        BgColor
        DAspect
        View
        AlphaScale  = 0.1
    end
    
    properties (Hidden, Transient)
        TextureSet = ...
            matlab.system.StringSet({'2D','3D'});
        SlicePlaneSet = ...
            matlab.system.StringSet({'XY','YZ'});
    end
    
    properties(DiscreteState)
        
    end
    
    % Pre-computed constants
    properties(Access = private)
        hSurfX
        hSurfY
        hSurfZ
    end
    
    methods
        function obj = VolumetricDataVisualizer(varargin)
            setProperties(obj,nargin,varargin{:})
        end
    end
    
    methods(Access = protected)
        
        function flag = isInactivePropertyImpl(obj,propertyName)
            
            if strcmp(propertyName,'SlicePlane')
                flag = ~strcmp(obj.Texture,'2D');
            elseif strcmp(propertyName,'XData') || ...
                    strcmp(propertyName,'YData') || ...
                    strcmp(propertyName,'ZData') || ...
                    strcmp(propertyName,'Alpha') || ...
                    strcmp(propertyName,'BgColor') || ...
                    strcmp(propertyName,'DAspect') || ...
                    strcmp(propertyName,'View') || ...
                    strcmp(propertyName,'AlphaScale')
                flag = ~strcmp(obj.Texture,'3D');
            else
                flag = false;
            end
            
        end
        
        function setupImpl(obj,u)
            
            vmin = obj.VRange(1);
            vmax = obj.VRange(2);
            cdata = (u-vmin)/(vmax-vmin); % in [0,1]
            if vmin < 0 % Signed case
                cdata = obj.Scale*(cdata-0.5)+0.5; % Scaled
            else % Unsigned case
                cdata = obj.Scale*cdata; % scaled;
            end
            %cdata(cdata<0)=0; % Clipping
            %cdata(cdata>1)=1;
            
            % ImageObject
            if isempty(obj.ImageObject)
                
                if strcmp(obj.Texture,'2D')
                    
                    if strcmp(obj.SlicePlane,'XY')
                        y = cdata(:,:,round(size(cdata,3)/2));
                    else
                        y = squeeze(cdata(:,round(size(cdata,2)/2),:));
                    end
                    obj.ImageObject = imshow(y);                    
                    
                else % 3D
                    
                    % Graphic group object
                    hVol = hggroup;
                                        
                    % Volume size
                    [height,width,depth] = size(cdata);
                    if isempty(obj.XData)
                        obj.XData = [0 width];
                    end
                    if isempty(obj.YData)
                        obj.YData = [0 height];
                    end
                    if isempty(obj.ZData)
                        obj.ZData = [0 depth];
                    end
                    
                    % Backgroud color
                    if isempty(obj.BgColor)
                        obj.BgColor = 'k';
                    end
                    
                    % DAspect
                    if isempty(obj.DAspect)
                        obj.DAspect = [1 1 1];
                    end
                    
                    % View
                    if isempty(obj.View)
                        obj.View = 3;
                    end
                    
                    % Options
                    obj.Opts = {...
                        'Parent',hVol,...
                        'cdatamapping',[],...
                        'alphadatamapping',[],...
                        'facecolor','texturemap',...
                        'edgealpha',0,...
                        'facealpha','texturemap'};
                    
                    if ndims(u) > 3
                        obj.Opts{4} = 'direct';
                    else
                        cdata = double(cdata);
                        obj.Opts{4} = 'scaled';
                    end
                    
                    if isempty(obj.Alpha)
                        if vmin < 0
                            alpha = 2*abs(cdata-0.5);
                        else
                            alpha = cdata;
                        end
                        if ndims(u) > 3
                            alpha = sqrt(sum(double(alpha).^2, 4));
                            alpha = alpha - min(alpha(:));
                            alpha = 1 - alpha / max(alpha(:));
                        end
                        obj.Opts{6} = 'scaled';
                    else
                        siz = size(cdata);
                        if ~isequal(siz(1:3), size(obj.Alpha))
                            error('Incorrect size of alphamatte');
                        end
                        alpha = obj.Alpha;
                        obj.Opts{6} = 'none';
                    end
                    
                    % Delete children
                    nChildren = length(hVol.Children);
                    for iChild = 1:nChildren
                        hVol.Children(iChild).delete()
                    end
                    
                    % Create z-slice
                    x = [obj.XData(1), obj.XData(2); obj.XData(1), obj.XData(2)];
                    y = [obj.YData(1), obj.YData(1); obj.YData(2), obj.YData(2)];
                    z = [obj.ZData(1), obj.ZData(1); obj.ZData(1), obj.ZData(1)];
                    diff = obj.ZData(2)-obj.ZData(1);
                    delta = diff/size(cdata,3);
                    %
                    nLays = size(cdata,3);
                    obj.hSurfZ = cell(nLays,1);
                    for n = 1:nLays
                        cslice = squeeze(cdata(:,:,n,:));
                        aslice = double(squeeze(alpha(:,:,n)));
                        obj.hSurfZ{n} = surface(x,y,z,cslice,'alphadata',aslice,...
                            ...'tag',sprintf('z%05d',n),...
                            obj.Opts{:});
                        z = z + delta;
                    end
                    
                    % Create x-slice
                    x = [obj.XData(1), obj.XData(1); obj.XData(1), obj.XData(1)];
                    y = [obj.YData(1), obj.YData(1); obj.YData(2), obj.YData(2)];
                    z = [obj.ZData(1), obj.ZData(2); obj.ZData(1), obj.ZData(2)];
                    diff = obj.XData(2)-obj.XData(1);
                    delta = diff/size(cdata,2);
                    %
                    nCols = size(cdata,2);
                    obj.hSurfX = cell(nCols,1);
                    for n = 1:nCols
                        cslice = squeeze(cdata(:,n,:,:));
                        aslice = double(squeeze(alpha(:,n,:)));
                        obj.hSurfX{n} = surface(x,y,z,cslice,'alphadata',aslice,...
                            ...'tag',sprintf('x%05d',n),...
                            obj.Opts{:});
                        x = x + delta;
                    end
                    
                    % Create y-slice
                    x = [obj.XData(1), obj.XData(1); obj.XData(2), obj.XData(2)];
                    y = [obj.YData(1), obj.YData(1); obj.YData(1), obj.YData(1)];
                    z = [obj.ZData(1), obj.ZData(2); obj.ZData(1), obj.ZData(2)];
                    diff = obj.YData(2)-obj.YData(1);
                    delta = diff/size(cdata,1);
                    %
                    nRows = size(cdata,1);
                    obj.hSurfY = cell(nRows,1);
                    for n = 1:nRows
                        cslice = squeeze(cdata(n,:,:,:));
                        aslice = double(squeeze(alpha(n,:,:)));
                        obj.hSurfY{n} = surface(x,y,z,cslice,'alphadata',aslice,...
                            ...'tag',sprintf('y%05d',n),...
                            obj.Opts{:});
                        y = y + delta;
                    end
                    
                    % Update CData
                    setappdata(hVol,'CData',cdata);
                    %
                    ax = hVol.Parent;
                    if vmin < 0 % Signed
                        alphamap(ax,[linspace(obj.AlphaScale, 0, 127) 0 linspace(0, obj.AlphaScale, 127)]);
                    else % Unsigned
                        alphamap(ax,[linspace(obj.AlphaScale, 0, 255) 0]);
                    end
                    %axis(ax,'off)
                    xlabel(ax,'X')
                    ylabel(ax,'Y')
                    zlabel(ax,'Z')
                    set(ax, 'Color', obj.BgColor)
                    view(ax,obj.View)
                    daspect(ax,obj.DAspect)
                    
                    % Update
                    obj.ImageObject = hVol;
                    
                end
                
            end
            
            % Colormap
            if isempty(obj.ColorMap)
                cmap = zeros(256,3);
                if strcmp(obj.Texture,'2D')
                    if obj.VRange(1) < 0 % Real
                        cmap(:,1) = [127:-1:0 zeros(1,128)].'/127; % R
                        cmap(:,2) = [zeros(1,128)    0:127].'/127; % G
                    else        % Non-negative
                        cmap = colormap(gray(256));
                    end
                else
                    cmap = zeros(256,3);
                    if obj.VRange(1) < 0 % Real
                        cmap(:,1) = [ones(128,1) ; zeros(128,1)]; % R
                        cmap(:,2) = [zeros(128,1) ; ones(128,1)]; % G
                    else        % Non-negative
                        cmap = colormap(bone(256));
                    end
                end
            else
                cmap = obj.ColorMap;
            end
            colormap(obj.ImageObject.Parent,cmap)
            
        end
        
        
        function imobj = stepImpl(obj,u)
            
            vmin = obj.VRange(1);
            vmax = obj.VRange(2);
            cdata = (u-vmin)/(vmax-vmin); % in [0,1]
            if vmin < 0 % Signed case
                cdata = obj.Scale*(cdata-0.5)+0.5; % Scaled
            else % Unsigned case
                cdata = obj.Scale*cdata; % scaled;
            end
            %cdata(cdata<0)=0; % Clipping
            %cdata(cdata>1)=1;
            
            if strcmp(obj.Texture,'2D')
                
                if strcmp(obj.SlicePlane,'XY')
                    y = cdata(:,:,round(size(cdata,3)/2));
                else
                    y = squeeze(cdata(:,round(size(cdata,2)/2),:));
                end
                obj.ImageObject.CData = y;
                
            else % obj.Texture = '3D'
                
                hVol = obj.ImageObject;
                
                if isempty(obj.Alpha)
                    if vmin < 0
                        alpha = 2*abs(cdata-0.5);
                    else
                        alpha = cdata;
                    end
                    if ndims(u) > 3
                        alpha = sqrt(sum(double(alpha).^2, 4));
                        alpha = alpha - min(alpha(:));
                        alpha = 1 - alpha / max(alpha(:));
                    end
                else
                    alpha = obj.Alpha;
                end
                
                % Update z-slice
                nLays = size(cdata,3);
                for n = 1:nLays
                    cslice = squeeze(cdata(:,:,n,:));
                    aslice = double(squeeze(alpha(:,:,n)));
                    obj.hSurfZ{n}.CData     = cslice;
                    obj.hSurfZ{n}.AlphaData = aslice;
                end
                
                % Update x-slice
                nCols = size(cdata,2);
                for n = 1:nCols
                    cslice = squeeze(cdata(:,n,:,:));
                    aslice = double(squeeze(alpha(:,n,:)));
                    obj.hSurfX{n}.CData     = cslice;
                    obj.hSurfX{n}.AlphaData = aslice;
                end
                
                % Update y-slice
                nRows = size(cdata,1);                
                for n = 1:nRows
                    cslice = squeeze(cdata(n,:,:,:));
                    aslice = double(squeeze(alpha(n,:,:)));
                    obj.hSurfY{n}.CData     = cslice;
                    obj.hSurfY{n}.AlphaData = aslice;
                end
                
                % Update CData
                setappdata(hVol,'CData',cdata);
                
            end
            
            imobj = obj.ImageObject;
            
        end
        
        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
    end
end
