classdef IdentityAnalysisSystemTestCase < matlab.unittest.TestCase
    %IDENTITYANALYSISSYSTEMTESTCASE Test case for IdentityAnalysisSystem
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
            
            classExpctd = 'saivdr.dictionary.AbstAnalysisSystem';
            
            import saivdr.dictionary.utility.*            
            target = IdentityAnalysisSystem();
            
            testCase.verifyTrue(isa(target,classExpctd));
            
        end
        
        function testStep(testCase,dim1,dim2,dim3)
            
            height = dim1;
            width  = dim2;
            depth  = dim3;
            srcImg = rand(height,width,depth);
            
            % Expected values
            coefExpctd = srcImg(:); 
            scalesExpctd = [ height width depth ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            target = IdentityAnalysisSystem();            
            
            % Actual values
            [ coefActual, scalesActual ] = step(target,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            testCase.verifyEqual(norm(coefActual(:)),norm(srcImg(:)),...
                'RelTol',1e-7,sprintf('Energy is not preserved.'));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifyEqual(scalesActual, scalesExpctd);
            
        end
        
        function testIsVectorizeFalse(testCase,dim1,dim2,dim3)
            
            height = dim1;
            width  = dim2;
            depth  = dim3;
            srcImg = rand(height,width,depth);
            
            % Expected values
            coefExpctd = srcImg; 
            scalesExpctd = [ height width depth ];
            
            % Instantiation of target class
            import saivdr.dictionary.utility.*            
            target = IdentityAnalysisSystem('IsVectorize',false);            
            
            % Actual values
            [ coefActual, scalesActual ] = step(target,srcImg);
            
            % Evaluation
            testCase.verifySize(coefActual,size(coefExpctd));
            testCase.verifyEqual(norm(coefActual(:)),norm(srcImg(:)),...
                'RelTol',1e-7,sprintf('Energy is not preserved.'));
            diff = max(abs(coefExpctd(:) - coefActual(:))./abs(coefExpctd(:)));
            testCase.verifyEqual(coefActual,coefExpctd,'RelTol',1e-7,sprintf('%g',diff));
            testCase.verifyEqual(scalesActual, scalesExpctd);
        end
    end
end