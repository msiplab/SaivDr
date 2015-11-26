classdef AbstAnalysisSystem < matlab.System %#codegen
    % ABSTANALYSISSYSTEM Abstract class of analysis system
    %
    % SVN identifier:
    % $Id: AbstAnalysisSystem.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2014-2015, Shogo MURAMATSU
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
    
   methods (Access = protected, Sealed = true)
        
        function N = getNumInputsImpl(~)
            % Specify number of System inputs
            N = 2; % Because stepImpl has one argument beyond obj
        end
        
        function N = getNumOutputsImpl(~)
            % Specify number of System outputs
            N = 2; % Because stepImpl has one output
        end
   end
    
end

