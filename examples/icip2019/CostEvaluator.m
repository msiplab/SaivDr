classdef CostEvaluator < matlab.System
    % Untitled Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    % Public, tunable properties
    properties (Nontunable)
        Observation
        MeasureProcess
        RefIdx2Ref
        OutputMode = 'Function'
    end
    
    properties (Nontunable, Logical)
        UseGpu = false
    end
    
    properties (Hidden, Transient)
        OutputModeSet = ...
            matlab.system.StringSet({'Function','Gradient'});
    end
    
    properties(DiscreteState)

    end

    % Pre-computed constants
    properties(Access = private)
         refIdx2RefGrad 
    end
    
    methods
        function obj = CostEvaluator(varargin)
            setProperties(obj,nargin,varargin{:})
            if isempty(obj.RefIdx2Ref)
               obj.RefIdx2Ref = RefractIdx2Reflect('OutputMode','Function','UseGpu',obj.UseGpu);
            else
               obj.RefIdx2Ref.release();  
               obj.RefIdx2Ref.OutputMode = 'Function';
            end
            if strcmp(obj.OutputMode,'Gradient')
                obj.refIdx2RefGrad = clone(obj.RefIdx2Ref);
                obj.refIdx2RefGrad.release();
                obj.refIdx2RefGrad.OutputMode = 'Gradient';
            end
        end
    end

    methods(Access = protected)
        
          
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.refIdx2RefGrad = obj.refIdx2RefGrad;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.refIdx2RefGrad = s.refIdx2RefGrad;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
        end
        
        function f = stepImpl(obj,u)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            %
            % y = Pƒ³(u)-v
            y = obj.MeasureProcess.step(obj.RefIdx2Ref.step(u),'Forward')...
                -obj.Observation;
            if strcmp(obj.OutputMode,'Function')
                % f = (1/2)||y||_2^2
                f = (1/2)*norm(y(:),2)^2;
            else
                % r = P'(Pƒ³(u)-v)
                r = obj.MeasureProcess.step(y,'Adjoint');
                % Þf = (dƒ³/du(u))r
                f = obj.refIdx2RefGrad.step(u,r);
            end
        end
        
        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
    end
end
