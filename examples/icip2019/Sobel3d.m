
classdef Sobel3d < matlab.System % codegen
   % Untitled4 Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    % Public, tunable properties
    properties (Nontunable)
        Kernel
        KernelMode = 'Normal'
        LambdaMax
    end
    
    properties (Nontunable, Logical)
        UseGpu = false
    end
    
    properties (Hidden, Transient)
        KernelModeSet = ...
            matlab.system.StringSet({'Normal','Absolute'});
        
    end
    
    properties(DiscreteState)

    end

    % Pre-computed constants
    properties(Access = private)

    end

    
    methods
        function obj = Sobel3d(varargin)
            setProperties(obj,nargin,varargin{:});
            %
            kernelxy = kron([ 1 2 1 ].', [ 1 2 1 ]);
            kernelz  = permute([ 1 0 -1 ].',[ 2 3 1 ]);
            obj.Kernel = convn(kernelxy,kernelz)/32;
            if strcmp(obj.KernelMode,'Absolute')
                obj.Kernel = abs(obj.Kernel);
            end            
            %
            obj.LambdaMax = sum(abs(obj.Kernel(:)))^2; 
        end
            
    end
    
    methods(Access = protected)
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
        end

        function y = stepImpl(obj,u)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            if obj.UseGpu
                u = gpuArray(u);
            end
            y = imfilter(u,obj.Kernel,'conv','circ');
            if obj.UseGpu
                y = gather(y);
            end
        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
    end
end

