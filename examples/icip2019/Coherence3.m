classdef Coherence3 < matlab.System
    % Untitled Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    % Public, tunable properties
    properties (Nontunable)
        Scale     = 1.0;
        Sigma     = 8.0;
        Frequency = 1/4;
        Kernel
    end
    
    properties (Nontunable, Logical)
        UseGpu = false
    end

    properties(DiscreteState)

    end

    % Pre-computed constants
    properties(Access = private)

    end

    methods
        function obj = Coherence3(varargin)
            setProperties(obj,nargin,varargin{:});
            %
            len = 2*round(5*obj.Sigma)+1; % Šï”‚ÉÝ’è
            n = -floor(len/2):floor(len/2);
            gc = obj.Scale...
                *exp(-n.^2./(2*obj.Sigma.^2)).*cos(2*pi*obj.Frequency*n);
            obj.Kernel = permute(gc(:),[2 3 1]);            
        end
    end
    
    methods(Access = protected)
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
        end

        function y = stepImpl(obj,u,dir)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            if obj.UseGpu
                u = gpuArray(u);
            end
            if strcmp(dir,'Forward')
                y = imfilter(u,obj.Kernel,'conv','circ');
            elseif strcmp(dir,'Adjoint')
                y = imfilter(u,obj.Kernel,'corr','circ');
            end
            if obj.UseGpu
                y = gather(y);
            end
        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
    end
end
