classdef VolumetricDataPlot < matlab.System
    % Untitled Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.
    
    % Public, tunable properties
    properties
        Direction = 'Z'
        Scales     = 1
        PlotObjects
    end
    
    properties (Nontunable, PositiveInteger)
        NumPlots
    end
    
    properties (Hidden, Transient)
        DirectionSet = ...
            matlab.system.StringSet({'X','Y','Z'});
    end
    
    properties(DiscreteState)
        
    end
    
    % Pre-computed constants
    properties(Access = private)
        
    end
    
    methods
        function obj = VolumetricDataPlot(varargin)
            setProperties(obj,nargin,varargin{:})     
            if isempty(obj.NumPlots)
                obj.NumPlots = 1;
            end
        end
    end
    
    methods(Access = protected)
        
        function setupImpl(obj,varargin)
            % Perform one-time calculations, such as computing constants
            
            obj.PlotObjects = cell(1,obj.NumPlots);
            % PlotObjects
            for iPlot = 1:obj.NumPlots
                if isempty(obj.PlotObjects{iPlot})
                    u = varargin{iPlot};
                    if strcmp(obj.Direction,'X')
                        y = u(round(size(u,1)/2),:,round(size(u,3)/2));
                    elseif strcmp(obj.Direction,'Y')
                        y = u(:,round(size(u,2)/2),round(size(u,3)/2));
                    else
                        y = u(round(size(u,1)/2),round(size(u,2)/2),:);
                    end
                    if isscalar(obj.Scales)
                        y = obj.Scales*squeeze(y);                        
                    else
                        y = obj.Scales(iPlot)*squeeze(y);                        
                    end
                    
                    obj.PlotObjects{iPlot} = plot(y);
                end
                hold on
            end
            hold off
            
        end
        
        function varargout = stepImpl(obj,varargin)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            varargout = cell(1,obj.NumPlots);
            for iPlot = 1:obj.NumPlots
                u = varargin{iPlot};
                if strcmp(obj.Direction,'X')
                    y = u(round(size(u,1)/2),:,round(size(u,3)/2));
                elseif strcmp(obj.Direction,'Y')
                    y = u(:,round(size(u,2)/2),round(size(u,3)/2));
                else
                    y = u(round(size(u,1)/2),round(size(u,2)/2),:);
                end
                if isscalar(obj.Scales)
                    y = obj.Scales*squeeze(y);
                else
                    y = obj.Scales(iPlot)*squeeze(y);
                end
                obj.PlotObjects{iPlot}.YData = y;
                varargout{iPlot} = obj.PlotObjects{iPlot};
            end
        end
        
        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
        
        function N = getNumInputsImpl(obj)
            N = obj.NumPlots;
        end
   
        function N = getNumOutputsImpl(obj)
            N = obj.NumPlots;
        end
        
    end
end