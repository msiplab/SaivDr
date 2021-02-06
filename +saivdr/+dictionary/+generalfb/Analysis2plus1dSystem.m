classdef Analysis2plus1dSystem < saivdr.dictionary.AbstAnalysisSystem
    %ANALYSIS2PLUS1DSYSTEM (2+1)-D analysis system
    %
    % Reference:
    %   Shogo Muramatsu and Hitoshi Kiya,
    %   ''Parallel Processing Techniques for Multidimensional Sampling
    %   Lattice Alteration Based on Overlap-Add and Overlap-Save Methods,''
    %   IEICE Trans. on Fundamentals, Vol.E78-A, No.8, pp.939-943, Aug. 1995
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2021, Shogo MURAMATSU
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
    properties (Access = protected, Constant = true)
        DATA_DIMENSION = 3
    end

    properties (Nontunable)
        AnalysisFiltersInXY = 1
        AnalysisFiltersInZ = 1
        DecimationFactor = [2 2 2]
        BoundaryOperation = 'Circular'
        FilterDomain = 'Spatial'
    end
    
    properties (Hidden, Transient)
        FilterDomainSet = ...
            matlab.system.StringSet({'Spatial'}); % 'Frequency' is not yet supported
    end
    
    properties (Nontunable, PositiveInteger)
        NumberOfLevelsInXY = 1
    end
    
    properties (Access = private, Nontunable)
        analysis2dSystemInXY
        analysis1dSystemInZ
    end

    methods
        % Constractor
        function obj = Analysis2plus1dSystem(varargin)
            setProperties(obj,nargin,varargin{:})

            import saivdr.dictionary.generalfb.Analysis3dSystem
            import saivdr.dictionary.utility.Direction                        
            % Instantiation of Analysis3dSystem for XY
            nDecsInXY = ...
                [ obj.DecimationFactor(Direction.VERTICAL:Direction.HORIZONTAL) 1 ]; 
            analysisFiltersInXY = permute(obj.AnalysisFiltersInXY,[1,2,4,3]);     
            obj.analysis2dSystemInXY = Analysis3dSystem(...
                'DecimationFactor',nDecsInXY,...
                'AnalysisFilters',analysisFiltersInXY,...
                'NumberOfLevels',obj.NumberOfLevelsInXY);
            % Instantiation of Analysis3dSystem for Z            
            nDecsInZ = [ 1 1 obj.DecimationFactor(Direction.DEPTH) ]; 
            analysisFiltersInZ = permute(obj.AnalysisFiltersInZ,[3,4,1,2]);
            obj.analysis1dSystemInZ = Analysis3dSystem(...
                'DecimationFactor',nDecsInZ,...
                'AnalysisFilters',analysisFiltersInZ,...
                'NumberOfLevels',1);
            
        end
    end

    methods (Access = protected)
        
        function setupImpl(~,~)
        end

        function [coefs,scales] = stepImpl(obj,srcImg)
            nChsInZ = size(obj.AnalysisFiltersInZ,2);
            % Analyze in Z
            [coefsInZ,scalesInZ] = ...
                obj.analysis1dSystemInZ.step(srcImg); 
            % Analyze in XY
            scales = [];
            coefs = [];
            sidx = 1;            
            for iChInZ = 1:nChsInZ
                % Get shape
                shape = scalesInZ(iChInZ,:);
                eidx = sidx + prod(shape) - 1;
                % Reshape subband images
                subImgInZ = reshape(coefsInZ(sidx:eidx),shape);
                % Analyze in XY
                [subCoefs,subScales] = ...
                    obj.analysis2dSystemInXY.step(subImgInZ);      
                %
                coefs = [ coefs subCoefs ];
                scales = [ scales; subScales ];
                %
                sidx = eidx + 1;
            end

        end

    end

    methods (Access = private)
    end

    methods (Access = private, Static = true)
    end

end
