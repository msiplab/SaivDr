classdef GaussianDenoiserBm4d < saivdr.restoration.denoiser.AbstGaussianDenoiseSystem % codegen
    % GAUSSIANDENOIZERBM4D Gaussian denoizer with soft thresholding
    %
    % Requirements: MATLAB R2019b
    %
    % Copyright (c) 2019-, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    
    % Public, tunable properties
    properties (Nontunable)
        IsVerbose = false;
        DoWiener  = true;        
    end
    
    properties (Access = private)

    end
    
    properties(DiscreteState)
        
    end

    
    methods
        function obj = GaussianDenoiserBm4d(varargin)
            setProperties(obj,nargin,varargin{:});
            if exist('bm4d','dir') ~= 7 || exist('bm4d','file') ~= 2
                disp('Downloading and unzipping BM4D_v3p2.zip...')
                url = 'http://www.cs.tut.fi/%7Efoi/GCF-BM3D/BM4D_v3p2.zip';
                disp(url)
                unzip(url,'./bm4d')
                addpath('./bm4d')                
            elseif exist('bm4d','dir') == 7 && exist('bm4d','file') ~= 2
                disp('Adding ./bm4d to path.')                
                addpath('./bm4d')
            elseif obj.IsVerbose
                disp('./bm4d already exists.')
                fprintf('See %s\n', ...
                    'http://www.cs.tut.fi/~foi/GCF-BM3D/index.html');
            end
            
        end
        
    end
    
    methods(Access = protected)
        

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.restoration.denoiser.AbstPlugGaussianDenoiseSystem(obj);
            %s.Synthesizer = matlab.System.saveObject(obj.Synthesizer);
            %s.threshold = obj.threshold;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            %obj.threshold = s.threshold;
            %obj.Synthesizer = matlab.System.loadObject(s.Synthesizer);
            loadObjectImpl@saivdr.restoration.denoiser.AbstPlugGaussianDenoiseSystem(obj,s,wasLocked);
        end
        
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
        end
        
        function y = stepImpl(obj,u)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            y = bm4d(u,'Gauss',obj.Sigma,'np',obj.DoWiener,obj.IsVerbose);
        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
    end
end

