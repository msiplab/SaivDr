classdef ParameterMatrixSet < saivdr.dictionary.utility.ParameterMatrixContainer
    %PARAMETERMATRIXSET Parameter matrix set (deprecated)
    %
    % This class was renamed ParameterMatrixContainer.
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2016, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
    %
    
    methods
        
        function obj = ParameterMatrixSet(varargin)
            obj = obj@saivdr.dictionary.utility.ParameterMatrixContainer(varargin{:});
            warning('This class is deprecated.')
        end
        
    end

end
