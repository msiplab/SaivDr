classdef RefractIdx2Reflect < matlab.System % codegen
    % Untitled4 Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.
    
    % Public, tunable properties
    properties (Nontunable)
        PhiMode = 'Reflection'
        VRange  = [ 1.0 1.5 ]
       OutputMode = 'Function'
    end
    
    properties (Nontunable, Logical)
       UseGpu = false
    end
    
    properties (Hidden, Transient)
        PhiModeSet = ...
            matlab.system.StringSet(...
            {'Reflection','Linear','Signed-Quadratic','Identity'});
        OutputModeSet = ...
            matlab.system.StringSet({'Function','Jacobian','Gradient'});
    end
    
    properties (Access = private)
    end
    
    properties(DiscreteState)
        
    end
    
    % Pre-computed constants
    properties(Access = private)
        dltFcn
        adtFcn
    end
    
    
    methods
        function obj = RefractIdx2Reflect(varargin)
            setProperties(obj,nargin,varargin{:});
            %
            obj.dltFcn = Sobel3d('KernelMode','Normal','UseGpu',obj.UseGpu);
            obj.adtFcn = Sobel3d('KernelMode','Absolute','UseGpu',obj.UseGpu);
        end
        
    end
    
    methods(Access = protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@matlab.System(obj);
            % Save the child System objects            
            s.dltFcn = matlab.System.saveObject(obj.dltFcn);
            s.adtFcn = matlab.System.saveObject(obj.adtFcn);
            % Save the protected & private properties           
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load protected and private properties            
            % Call base class method to load public properties            
            % Load the child System objects            
            obj.dltFcn = matlab.System.loadObject(s.dltFcn);            
            obj.adtFcn = matlab.System.loadObject(s.adtFcn);
            %
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        
        %function processTunedPropertiesImpl(obj)
        %    propChange = isChangedProperty(obj,'OutputMode');
        %    if propChange
        %    end
        %end
        
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
        end
        
        function y = stepImpl(obj,varargin)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            arrayU = varargin{1};
            vmin   = obj.VRange(1);
            vmax   = obj.VRange(2);
            %
            if strcmp(obj.OutputMode,'Function')
                
                if strcmp(obj.PhiMode,'Linear')
                    
                    beta1  = 2*abs(vmax-vmin)/(vmax+vmin)^2;
                    y = -beta1*obj.dltFcn.step(arrayU);
                    
                elseif strcmp(obj.PhiMode,'Signed-Quadratic')
                    
                    beta2   = 4/(vmax+vmin)^2;
                    arrayDltU = obj.dltFcn.step(arrayU);
                    y = -beta2*abs(arrayDltU).*arrayDltU;
                    
                elseif strcmp(obj.PhiMode,'Reflection')
                    
                    arrayDltU = obj.dltFcn.step(arrayU);
                    arrayAddU = obj.adtFcn.step(arrayU);
                    y = -(1./(arrayAddU.*arrayAddU)).*abs(arrayDltU).*arrayDltU;
                    
                else
                    
                    y = arrayU;
                    
                end
                
            elseif strcmp(obj.OutputMode,'Jacobian')
                
                nCols = numel(arrayU);
                arrayD = eye(nCols);
                
                if strcmp(obj.PhiMode,'Linear')
                    
                    beta1   = 2*abs(vmax-vmin)/(vmax+vmin)^2;
                    for iCol = 1:nCols
                        mask = zeros(size(arrayU));
                        mask(iCol) = 1;
                        vecD = beta1*obj.dltFcn.step(mask);
                        arrayD(:,iCol) = vecD(:);
                    end
                    
                elseif strcmp(obj.PhiMode,'Signed-Quadratic')
                    
                    arrayDltU = obj.dltFcn.step(arrayU);
                    beta2     = 4/(vmax+vmin)^2;
                    for iCol = 1:nCols
                        mask = zeros(size(arrayU));
                        mask(iCol) = 1;
                        vecD = 2*beta2*obj.dltFcn.step(mask.*abs(arrayDltU)); % ”ºì—p‘f‚Í•„†”½“]
                        arrayD(:,iCol) = vecD(:);
                    end
                    
                elseif strcmp(obj.PhiMode,'Reflection')
                    
                    arrayDltU = obj.dltFcn.step(arrayU);
                    arrayAddU = obj.adtFcn.step(arrayU);
                    arrayT = abs(arrayDltU)./(arrayAddU.^3);
                    for iCol = 1:nCols
                        mask = zeros(size(arrayU));
                        mask(iCol) = 1;
                        vecD = 2*( obj.dltFcn.step(mask.*arrayAddU.*arrayT)... % ”ºì—p‘f‚Í•„†”½“]
                            + obj.adtFcn.step(mask.*arrayDltU.*arrayT) );      % ”ºì—p‘f‚Í‚»‚Ì‚Ü‚Ü
                        arrayD(:,iCol) = vecD(:);
                    end
                    
                end
                y = arrayD;
                
            else % if strcmp(obj.OutputMode,'Gradient')
                
                arrayR = varargin{2};
               
                if strcmp(obj.PhiMode,'Linear')
                    
                    beta1  = 2*abs(vmax-vmin)/(vmax+vmin)^2;
                    arrayD = beta1*obj.dltFcn.step(arrayR);
                    
                elseif strcmp(obj.PhiMode,'Signed-Quadratic')
                    
                    arrayDltU = obj.dltFcn.step(arrayU);
                    beta2     = 4/(vmax+vmin)^2;
                    arrayD = 2*beta2*obj.dltFcn.step(abs(arrayDltU).*arrayR); 
                    
                elseif strcmp(obj.PhiMode,'Reflection')
                    
                    arrayDltU = obj.dltFcn.step(arrayU);
                    arrayAddU = obj.adtFcn.step(arrayU);
                    arrayT = abs(arrayDltU)./(arrayAddU.^3);
                    arrayT = arrayT.*arrayR;
                    arrayD = 2*( obj.dltFcn.step(arrayAddU.*arrayT) ...
                        + obj.adtFcn.step(arrayDltU.*arrayT) );
                else
                    
                    arrayD = arrayR;

                end
                y = arrayD;
                
            end
            
        end
        
        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
        
        function n = getNumInputsImpl(obj)
            if strcmp(obj.OutputMode,'Gradient')
                n = 2;
            else
                n = 1;
            end
        end
    end
end

