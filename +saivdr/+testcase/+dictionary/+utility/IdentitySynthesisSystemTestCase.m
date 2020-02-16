classdef IdentitySynthesisSystemTestCase < matlab.unittest.TestCase
    %IDENTITYSYNTHESISSYSTEMTESTCASE Test case for IdentitySynthesisSystem
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
       
    properties (TestParameter)
        %type = {'single','double','uint16'};
        dim1 = struct('small',8, 'medium', 16, 'large', 32);
        dim2 = struct('small',8, 'medium', 16, 'large', 32);
        dim3 = struct('small',8, 'medium', 16, 'large', 32);
    end
    
    methods (Test)
        
        function testConstruction(testCase)
            
            classExpctd = 'saivdr.dictionary.AbstSynthesisSystem';
            
            import saivdr.dictionary.utility.*            
            target = IdentitySynthesisSystem();
            
            testCase.verifyTrue(isa(target,classExpctd));
            
        end
        
        function testStep(testCase,dim1,dim2,dim3)
            
            height = dim1;
            width  = dim2;
            depth  = dim3;
            imgExpctd = rand(height,width,depth);
            dimExpctd = [ height width depth ];
            scalesExpctd = dimExpctd;
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            target = IdentitySynthesisSystem();
            
            % Actual values
            subCoefs = imgExpctd(:);
            imgActual = step(target,subCoefs,scalesExpctd);
            
            % Evaluation
            testCase.verifySize(imgActual,dimExpctd);
            diff = max(abs(imgExpctd(:) - imgActual(:))./abs(imgExpctd(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'RelTol',1e-7,sprintf('%g',diff));
            
        end
        
    end
end
    