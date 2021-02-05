classdef Synthesis2plus1dSystem < saivdr.dictionary.AbstSynthesisSystem
    %SYNTHESIS2plus1DSYSTEM (2+1)-D synthesis system
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
    properties (Nontunable)
        SynthesisFiltersInXY = 1
        SynthesisFiltersInZ = 1
        DecimationFactor = [2 2 2]
        BoundaryOperation = 'Circular'
        FilterDomain = 'Spatial'
    end

    properties (Access = private, Nontunable)
        synthesis2dSystemInXY
        synthesis1dSystemInZ
    end
    
    methods

        % Constractor
        function obj = Synthesis2plus1dSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            
            import saivdr.dictionary.generalfb.Synthesis3dSystem
            import saivdr.dictionary.utility.Direction            
            % Instantiation of Synthesis3dSystem for XY
            nDecsInXY = ...
                [ obj.DecimationFactor(Direction.VERTICAL:Direction.HORIZONTAL) 1 ];
            synthesisFiltersInXY = permute(obj.SynthesisFiltersInXY,[1,2,4,3]);
            obj.synthesis2dSystemInXY = Synthesis3dSystem(...
                'DecimationFactor',nDecsInXY,...
                'SynthesisFilters',synthesisFiltersInXY);
            % Instantiation of Synthesis3dSystem for Z
            nDecsInZ = [ 1 1 obj.DecimationFactor(Direction.DEPTH) ];
            synthesisFiltersInZ = permute(obj.SynthesisFiltersInZ,[3,4,1,2]);
            obj.synthesis1dSystemInZ = Synthesis3dSystem(...
                'DecimationFactor',nDecsInZ,...
                'SynthesisFilters',synthesisFiltersInZ);            
        end

        function setFrameBound(obj,frameBound)
            obj.FrameBound = frameBound;
        end

    end
    
    
    methods (Access = protected)
        
        function setupImpl(~,~,~)            
        end
            
        function recImg = stepImpl(obj,coefs,scales)
            nChsInZ = size(obj.SynthesisFiltersInZ,2);
            nChsInXY = size(obj.SynthesisFiltersInXY,3); % TODO: extension to multilevel
            % Synthesize in XY
            sidx = 1;
            scalesInZ = [];
            coefsInZ = [];
            for iChInZ = 1:nChsInZ
                % Get Coefs. and scales
                subScales = scales((iChInZ-1)*nChsInXY+1:iChInZ*nChsInXY,:); 
                eidx = sidx + sum(prod(subScales,2))-1;
                subCoefs = coefs(sidx:eidx);
                subImgInZ = ...
                    obj.synthesis2dSystemInXY.step(subCoefs,subScales);      
                % Prepare Coefs. and scales for synthesizing in Z
                scalesInZ = [ scalesInZ; size(subImgInZ) ];
                coefsInZ = [ coefsInZ subImgInZ(:).' ];
                %
                sidx = eidx + 1;
            end
            % Synthesize in Z
            recImg = ...
                obj.synthesis1dSystemInZ.step(coefsInZ,scalesInZ);
        end
        
    end
    
end
