classdef SubbandSpecificationTestCase < matlab.unittest.TestCase
    %SUBBANDSPECIFICATIONTESTCASE Test case for SubbandSpecification
    %
    % SVN identifier:
    % $Id: SubbandSpecificationTestCase.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
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
    % http://msiplab.eng.niigata-u.ac.jp/    
    %
    properties
        sbsp
    end
    
    methods (TestMethodTeardown)
     
        function deleteObject(testCase)
            delete(testCase.sbsp);
        end
        
    end
    
    methods (Test)
       
        % Test for default construction
        function testGetSubbandIndex(testCase)
            
            % Expected values
            sbidcsExpctd = [ 1 4 3 2 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification();
            
            % Actual values
            sbidcsActual = get(testCase.sbsp,'subbandIndices');
            
            % Evaluation
            testCase.verifyEqual(sbidcsActual, sbidcsExpctd);
        end
    
        function testGetSubbandIndex44(testCase)
            
            dec = [4 4];
            
            % Expected values
            sbidcsExpctd = ...
                [ 1 2 13 14 3 4 15 16 9 10 5 6 11 12 7 8 ];
             %   [ 1 2 5 6 11 12 15 16 9 10 13 14 3 4 7 8 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification(...
                'DecimationFactor',dec);
            
            % Actual values
            sbidcsActual = get(testCase.sbsp,'subbandIndices');
            
            % Evaluation
            testCase.verifyEqual(sbidcsActual,sbidcsExpctd);
        end

        % Test for default construction
        function testConstructor(testCase)
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            directionExpctd = Direction.VERTICAL;
            alphaExpctd = 0;
            matrixVExpctd = [
                0.5 0.0 ;
                0.0 0.5 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification();
            
            % Actual values
            directionActual = get(testCase.sbsp,'Direction');
            alphaActual   = get(testCase.sbsp,'Alpha');
            matrixVActual = get(testCase.sbsp,'matrixV');
            
            % Evaluation
            testCase.verifyEqual(directionActual,directionExpctd); 
            testCase.verifyEqual(alphaActual,alphaExpctd); 
            testCase.verifyEqual(matrixVActual,matrixVExpctd); 
        end

        % Test for construction with positive alpha
        function testConstructorWithPositiveAlpha(testCase)
            
            % Input value
            alpha = 1.0;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            directionExpctd = Direction.VERTICAL;
            alphaExpctd = alpha;
            matrixVExpctd = [
                0.5 0.5 ;
                0.0 0.5 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha);
            
            % Actual values
            directionActual = get(testCase.sbsp,'Direction');
            alphaActual   = get(testCase.sbsp,'Alpha');
            matrixVActual = get(testCase.sbsp,'matrixV');
            
            % Evaluation
            testCase.verifyEqual(directionActual,directionExpctd); 
            testCase.verifyEqual(alphaActual,alphaExpctd);
            testCase.verifyEqual(matrixVActual,matrixVExpctd); 
        end

        % Test for construction with negative alpha
        function testConstructorWithNegativeAlpha(testCase)
            
            % Input value
            alpha = -1.0;
            
            % Expected values
            import saivdr.dictionary.utility.Direction
            directionExpctd = Direction.VERTICAL;
            alphaExpctd = alpha;
            matrixVExpctd = [
                0.5 -0.5 ;
                0.0  0.5 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha);
            
            % Actual values
            directionActual = get(testCase.sbsp,'Direction');
            alphaActual = get(testCase.sbsp,'Alpha');
            matrixVActual = get(testCase.sbsp,'matrixV');
            
            % Evaluation
            testCase.verifyEqual(directionActual, directionExpctd);
            testCase.verifyEqual(alphaActual, alphaExpctd);
            testCase.verifyEqual(matrixVActual, matrixVExpctd);
        end

        % Test for construction with vertical direction
        function testConstructorWithVerticalDirection(testCase)
            
            % Input value
            import saivdr.dictionary.utility.Direction
            alpha = 1.0;
            direction = Direction.VERTICAL;
            
            % Expected values
            directionExpctd = direction;
            alphaExpctd = alpha;
            matrixVExpctd = [
                0.5 0.5 ;
                0.0 0.5 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction);
            
            % Actual values
            directionActual = get(testCase.sbsp,'Direction');
            alphaActual = get(testCase.sbsp,'Alpha');
            matrixVActual = get(testCase.sbsp,'matrixV');
            
            % Evaluation
            testCase.verifyEqual(directionActual, directionExpctd);
            testCase.verifyEqual(alphaActual, alphaExpctd);
            testCase.verifyEqual(matrixVActual, matrixVExpctd);
        end

        % Test for illeagal argument exception
        function testConstructorIlleagalArgumentException(testCase)
            
            % Invalid input
            alpha = 1.0;
            direction = 0;
            
            % Expectec values
            exceptionIdExpctd = 'SaivDr:IllegalArgumentException';
            messageExpctd = ...
                sprintf('%d: Direction must be either of 1 or 2.', ...
                direction);
            
            % Instantiation of target class
            try
                import saivdr.dictionary.nsgenlotx.design.*
                SubbandSpecification(...
                    'Alpha',alpha,...
                    'Direction',direction);
                testCase.verifyFail(sprintf('%s must be thrown.',...
                    exceptionIdExpctd));
            catch me
                exceptionIdActual = me.identifier;
                testCase.verifyEqual(exceptionIdActual, exceptionIdExpctd);
                messageActual = me.message;
                testCase.verifyEqual(messageActual, messageExpctd);
            end
        end

        % Test for construction with horizontal direction
        function testConstructorWithHorizontalDirection(testCase)
            %%% Horizontal direction
            % Input value
            import saivdr.dictionary.utility.Direction
            alpha = -1.0;
            direction = Direction.HORIZONTAL;
            
            % Expected values
            directionExpctd = direction;
            alphaExpctd = alpha;
            matrixVExpctd = [
                0.5 0.0 ;
                -0.5 0.5 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction);
            
            % Actual values
            directionActual = get(testCase.sbsp,'Direction');
            alphaActual = get(testCase.sbsp,'Alpha');
            matrixVActual = get(testCase.sbsp,'matrixV');
            
            % Evaluation
            testCase.verifyEqual(directionActual, directionExpctd);
            testCase.verifyEqual(alphaActual, alphaExpctd);
            testCase.verifyEqual(matrixVActual, matrixVExpctd);
        end
        

        % Test for default setting
        function testDefaultRegionOfSupport(testCase)
            
            % Input argument
            nPoints = [16 16];
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification(...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end

        % Test for default setting
        function testDefaultSmallRegionOfSupport(testCase)
            
            % Input argument
            nPoints = [8 8];
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 ;
                0 0 1 1 1 1 0 0 ;
                0 0 1 1 1 1 0 0 ;
                0 0 1 1 1 1 0 0 ;
                0 0 1 1 1 1 0 0 ;
                0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification(...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end
        
        % Test for vertical positive setting
        function testVerticalRosWithPositiveAlpha(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction
            nPoints = [16 16];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            
            % Expected array
            rosExpctd = [
                0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd,message);
        end
        
        % Test for vertical positive small setting
        function testVerticalSmallRosWithPositiveAlpha(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction
            nPoints = [8 8];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            
            % Expected array
            rosExpctd = [
                0 0 1 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 ;
                0 0 1 1 1 0 0 0 ;
                0 0 1 1 1 1 0 0 ;
                0 0 0 1 1 1 0 0 ;
                0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end

        % Test for vertical negative setting
        function testVerticalRosWithNegativeAlpha(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = -1.0;
            direction = Direction.VERTICAL;
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end
        
        % Test for horizontal positive setting
        function testHorizontalRosWithPositiveAlpha(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 ;
                0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 ;
                0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);            
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end


        % Test for horizontal negative setting
        function testHorizontalRosWithNegativeAlpha(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = -1.0; %1.0;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 ;
                0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 ;
                0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 ;
                0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 ;
                0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 ;
                0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);  
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end
        
        % Test for horizontal wrapping
        function testHorizontalWrappingRos(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = -2.0;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 ;
                1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 ;
                0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 ;
                0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 ;
                1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 ;
                1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);  
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end
        
        % Test for vertical positive setting
        function testVerticalWrappingRos(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 2.0;
            direction = Direction.VERTICAL;
            
            % Expected array
            rosExpctd = [
                0 0 0 0 1 1 1 0 0 0 0 1 0 0 0 0 ;
                0 0 0 0 1 1 1 0 0 0 0 1 0 0 0 0 ;
                0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 ;
                0 0 0 0 1 0 0 0 0 1 1 1 0 0 0 0 ;
                0 0 0 0 1 0 0 0 0 1 1 1 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);  
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end
        
        % Test for horizontal positive setting
        function testHorizontalRosWithTransition(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            transition = 0.25;
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','RegionOfSupport');
            set(testCase.sbsp,'Transition',transition);
            rosActual = step(testCase.sbsp,nPoints);
            
            % Observation
            %               subplot(1,2,1); subimage(rosExpctd);
            %               subplot(1,2,2); subimage(rosActual);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end

        % Test for vertical positive setting
        function testVerticalRosWithTransition(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            transition = 0.25;
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','RegionOfSupport');
            set(testCase.sbsp,'Transition',transition);
            rosActual = step(testCase.sbsp,nPoints);
            
            % Observation
            %              subplot(1,2,1); subimage(rosExpctd);
            %              subplot(1,2,2); subimage(rosActual);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end
        
        % Test for setting direction
        function testSetDirection(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            preDirection = Direction.VERTICAL;
            postDirection = Direction.HORIZONTAL;
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 ;
                0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 ;
                0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',preDirection,...
                'OutputMode','RegionOfSupport');
            set(testCase.sbsp, 'Direction',postDirection);
            rosActual = step(testCase.sbsp,nPoints);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end

        % Test for setting direction
        function testSetAlpha(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            preAlpha = -1.0;
            postAlpha = 1.0;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 ;
                0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 ;
                0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',preAlpha,...
                'Direction',direction,...
                'OutputMode','RegionOfSupport');
            set(testCase.sbsp, 'Alpha',postAlpha);
            rosActual = step(testCase.sbsp,nPoints);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end

        % Test for vertical positive setting
        function testGetSubbandLocation(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            
            % Expected array
            rosExpctd(:,:,1) = [
                0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,2) = [
                0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 ];
            
            rosExpctd(:,:,3) = [
                0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 ;
                1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 ;
                1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 ;
                1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 ;
                1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,4) = [
                1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 ;
                1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 ;
                1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 ;
                1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 ];
            rosExpctd = logical(rosExpctd);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','SubbandLocation');
            
            % Evaluation
            for idx = 1:4
                rosActual = step(testCase.sbsp,nPoints,idx);
                message = sprintf(...
                    '%d: Actual array is not an expected one.',idx);
                testCase.verifyEqual(rosActual,rosExpctd(:,:,idx), ...
                    message);
            end
            
        end

        % Test for vertical positive setting
        function testGetPassStopAssignmentWithSubbandIndex(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [ 16 16 ];
            
            % Expected assignment
            sbIdcsExpctd = [ 1 4 3 2 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'OutputMode','PassStopAssignment');
            
            % Evaluation
            sbIdcsActual = zeros(1,4);
            for idx = 1:4
                [~,sbIdcsActual(idx)] = ...
                    step(testCase.sbsp,nPoints,idx);
                testCase.verifyEqual(sbIdcsActual(idx), sbIdcsExpctd(idx));
            end
            
        end       

        % Test for vertical positive setting
        function testGetAmplitudeSpecificationWithSubbandIndex(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [ 16 16 ];
            
            % Expected assignment
            sbIdcsExpctd = [ 1 4 3 2 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'OutputMode','AmplitudeSpecification');
            
            % Evaluation
            sbIdcsActual = zeros(1,4);
            for idx = 1:4
                [~,sbIdcsActual(idx)] = ...
                    step(testCase.sbsp,nPoints,idx,[]);
                testCase.verifyEqual(sbIdcsActual(idx), sbIdcsExpctd(idx));
            end
            
        end

        % Test for vertical positive setting
        function testGetPassStopAssignment44WithSubbandIndex(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            dec = [4 4];
            nPoints = [ 16 16 ];
            
            % Expected assignment
            sbIdcsExpctd = ...
                [ 1 2 13 14 3 4 15 16 9 10 5 6 11 12 7 8 ];
             %   [ 1 2 5 6 11 12 15 16 9 10 13 14 3 4 7 8 ];
             
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'DecimationFactor',dec,...
                'OutputMode','PassStopAssignment');
            
            % Evaluation
            sbIdcsActual = zeros(1,16);
            for idx = 1:16
                [~,sbIdcsActual(idx)] = ...
                    step(testCase.sbsp,nPoints,idx);
                testCase.verifyEqual(sbIdcsActual(idx), sbIdcsExpctd(idx));
            end
            
        end
        
        % Test for vertical positive setting
        function testSubbandAssignment(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            transition = 0.25;
            direction = Direction.VERTICAL;
            
            % Expected array
            saExpctd = [
                4 4 4 0 0 0 2 2 2 2 2 0 0 0 4 4 ;
                4 4 4 0 0 0 0 2 2 2 2 0 0 0 0 4 ;
                4 4 4 0 0 1 0 0 2 2 2 0 0 3 0 0 ;
                0 4 4 0 0 1 1 0 0 2 2 0 0 3 3 0 ;
                0 0 4 0 0 1 1 1 0 0 2 0 0 3 3 3 ;
                3 0 0 0 0 1 1 1 1 0 0 0 0 3 3 3 ;
                3 3 0 0 0 1 1 1 1 1 0 0 0 3 3 3 ;
                3 3 3 0 0 1 1 1 1 1 1 0 0 3 3 3 ;
                3 3 3 0 0 0 1 1 1 1 1 0 0 0 3 3 ;
                3 3 3 0 0 0 0 1 1 1 1 0 0 0 0 3 ;
                3 3 3 0 0 2 0 0 1 1 1 0 0 4 0 0 ;
                0 3 3 0 0 2 2 0 0 1 1 0 0 4 4 0 ;
                0 0 3 0 0 2 2 2 0 0 1 0 0 4 4 4 ;
                4 0 0 0 0 2 2 2 2 0 0 0 0 4 4 4 ;
                4 4 0 0 0 2 2 2 2 2 0 0 0 4 4 4 ;
                4 4 4 0 0 2 2 2 2 2 2 0 0 4 4 4 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','SubbandAssignment');
            set(testCase.sbsp,'Transition',transition);
            saActual = step(testCase.sbsp,nPoints);
            
            % Observation
            %              map = [0 0 0 ; 1 1 1 ; 1 0 0 ; 0 1 0 ; 0 0 1];
            %              subplot(1,2,1); subimage(saExpctd+1,map);
            %              subplot(1,2,2); subimage(saActual+1,map);
            %
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(saActual, saExpctd, message);
            
        end

        % Test for dec = [4 4]
        function testConstructor44(testCase)
            
            % Expected 
            import saivdr.dictionary.utility.Direction
            dec = [4 4];
            directionExpctd = Direction.VERTICAL;
            alphaExpctd = 0;
            matrixVExpctd = [
                0.25 0.0 ;
                0.0 0.25 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'DecimationFactor',dec);
            
            % Actual values
            directionActual = get(testCase.sbsp,'Direction');
            alphaActual     = get(testCase.sbsp,'Alpha');
            matrixVActual   = get(testCase.sbsp,'matrixV');
            
            % Evaluation
            testCase.verifyEqual(directionActual, directionExpctd);
            testCase.verifyEqual(alphaActual, alphaExpctd);
            testCase.verifyEqual(matrixVActual, matrixVExpctd);
        end
                
        % Test for construction with positive alpha
        function testConstructor44WithPositiveAlpha(testCase)
            
            % Input value
            import saivdr.dictionary.utility.Direction        
            alpha = 1.0;
            dec = [4 4];
            
            % Expected values
            directionExpctd = Direction.VERTICAL;
            alphaExpctd = alpha;
            matrixVExpctd = [
                0.25 0.25 ;
                0.0 0.25 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'DecimationFactor',dec);
            
            % Actual values
            directionActual = get(testCase.sbsp,'Direction');
            alphaActual     = get(testCase.sbsp,'Alpha');
            matrixVActual   = get(testCase.sbsp,'matrixV');
            
            % Evaluation
            testCase.verifyEqual(directionActual, directionExpctd);
            testCase.verifyEqual(alphaActual, alphaExpctd);
            testCase.verifyEqual(matrixVActual, matrixVExpctd);
        end
        
        % Test for construction with negative alpha
        function testConstructor44WithNegativeAlpha(testCase)
            
            % Input value
            import saivdr.dictionary.utility.Direction        
            alpha = -1.0;
            dec = [4 4];
            
            % Expected values
            directionExpctd = Direction.VERTICAL;
            alphaExpctd = alpha;
            matrixVExpctd = [
                0.25 -0.25 ;
                0.0  0.25 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'DecimationFactor',dec);
            
            % Actual values
            directionActual = get(testCase.sbsp,'Direction');
            alphaActual     = get(testCase.sbsp,'Alpha');
            matrixVActual   = get(testCase.sbsp,'matrixV');
            
            % Evaluation
            testCase.verifyEqual(directionActual, directionExpctd);
            testCase.verifyEqual(alphaActual, alphaExpctd);
            testCase.verifyEqual(matrixVActual, matrixVExpctd);
        end
        
        % Test for construction with vertical direction
        function testConstructor44WithVerticalDirection(testCase)
            
            % Input value
            import saivdr.dictionary.utility.Direction        
            alpha = 1.0;
            direction = Direction.VERTICAL;
            dec = [4 4];
            
            % Expected values
            directionExpctd = direction;
            alphaExpctd = alpha;
            matrixVExpctd = [
                0.25 0.25 ;
                0.0 0.25 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp= SubbandSpecification(...
                'Alpha',alpha,....
                'Direction',direction,...
                'DecimationFactor',dec);
            
            % Actual values
            directionActual = get(testCase.sbsp,'Direction');
            alphaActual     = get(testCase.sbsp,'Alpha');
            matrixVActual   = get(testCase.sbsp,'matrixV');
            
            % Evaluation
            testCase.verifyEqual(directionActual, directionExpctd);
            testCase.verifyEqual(alphaActual, alphaExpctd);
            testCase.verifyEqual(matrixVActual, matrixVExpctd);
        end
        
        % Test for construction with horizontal direction
        function testConstructor44WithHorizontalDirection(testCase)
            %%% Horizontal direction
            % Input value
            import saivdr.dictionary.utility.Direction        
            alpha = -1.0;
            direction = Direction.HORIZONTAL;
            dec = [4 4];
            
            % Expected values
            directionExpctd = direction;
            alphaExpctd = alpha;
            matrixVExpctd = [
                0.25 0.0 ;
                -0.25 0.25 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec);
            
            % Actual values
            directionActual = get(testCase.sbsp,'Direction');
            alphaActual = get(testCase.sbsp,'Alpha');
            matrixVActual = get(testCase.sbsp,'matrixV');
            
            % Evaluation
            testCase.verifyEqual(directionActual, directionExpctd);
            testCase.verifyEqual(alphaActual, alphaExpctd);
            testCase.verifyEqual(matrixVActual, matrixVExpctd);
        end

        % Test for default setting
        function testRegionOfSupport44(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            dec = [4 4];
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'DecimationFactor',dec,...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end

        % Test for default setting
        function testSmallRegionOfSupport44(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [8 8];
            dec = [4 4];
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 ;
                0 0 0 1 1 0 0 0 ;
                0 0 0 1 1 0 0 0 ;
                0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'DecimationFactor',dec,...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end
        
        % Test for vertical positive setting
        function testVerticalRosWithPositiveAlpha44(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            dec = [ 4 4 ];
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec,...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end

        % Test for horizontal negative setting
        function testHorizontalRosWithNegativeAlpha44(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = -1.0; %1.0;
            direction = Direction.HORIZONTAL;
            dec = [4 4];
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec,...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
        end

        function testGetSubbandLocation44(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            dec = [4 4];
            
            % Expected array
            rosExpctd(:,:,1) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,2) = [
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,3) = [
                0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,4) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,5) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,6) = [
                0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,7) = [
                0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 ];
            
            rosExpctd(:,:,8) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,9) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,10) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,11) = [
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ];
            
            rosExpctd(:,:,12) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,13) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,14) = [
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,15) = [
                0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 0 ];
            
            rosExpctd(:,:,16) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            rosExpctd = logical(rosExpctd);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            spfb = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec,...
                'OutputMode','SubbandLocation');
            
            % Evaluation
            for idx = 1:size(rosExpctd,3)
                rosActual = step(spfb,nPoints,idx);
                message = sprintf(...
                    'subband%02d: Actual array is not an expected one.',idx);
                testCase.verifyEqual(...
                    rosActual, rosExpctd(:,:,idx),message);
            end
        end
        
        % Test for vertical positive setting
        function testVerticalRosWithPositiveAlpha88(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            dec = [ 8 8 ];
            
            % Expected array
            rosExpctd = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec,...
                'OutputMode','RegionOfSupport');
            rosActual = step(testCase.sbsp,nPoints);
            
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(rosActual, rosExpctd, message);
            
        end
        
        % Test for vertical positive setting
        function testGetAmplitudeSpecification(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            
            % Expected array
            asExpctd = [8.43479777133728e-65,0.00843507643138303,0.109375000000000,0.350939923568617,0.500000000000000,0.350939923568617,0.109375000000000,0.00843507643138303,8.43479777133728e-65,0.00843507643138301,0.109375000000000,0.350939923568616,0.500000000000000,0.350939923568617,0.109375000000000,0.00843507643138303;8.39907237528441e-65,0.00847095497043863,0.115624714595798,0.427637966981143,0.773001901487318,0.773001901487318,0.427637966981143,0.115624714595798,0.00847095497043863,8.39907237528441e-65,0.00797914604942120,0.0897582934725953,0.226998098512681,0.226998098512682,0.0897582934725955,0.00797914604942122;7.94508807444975e-65,0.00843507643138303,0.116116523516816,0.452073306354487,0.941941738241592,1.19506387940602,0.941941738241592,0.452073306354487,0.116116523516816,0.00843507643138303,7.94508807444975e-65,0.00654806429956250,0.0580582617584078,0.103056273456746,0.0580582617584078,0.00654806429956252;6.52011471590472e-65,0.00797914604942122,0.115624714595798,0.453996197025363,0.995764522514781,1.45624550950204,1.45624550950204,0.995764522514781,0.453996197025363,0.115624714595798,0.00797914604942122,6.52011471590472e-65,0.00423547748521930,0.0263582300442202,0.0263582300442202,0.00423547748521931;4.21739888566864e-65,0.00654806429956251,0.109375000000000,0.452073306354487,1,1.53945573867507,1.77450847648318,1.53945573867507,1,0.452073306354487,0.109375000000000,0.00654806429956251,4.21739888566864e-65,0.00192289067087611,0.00674152351681560,0.00192289067087611;1.91468305543257e-65,0.00423547748521932,0.0897582934725954,0.427637966981143,0.995764522514781,1.54600380297464,1.87590433043376,1.87590433043376,1.54600380297464,0.995764522514781,0.427637966981143,0.0897582934725954,0.00423547748521932,1.91468305543257e-65,0.000491808921017411,0.000491808921017411;4.89709696887535e-66,0.00192289067087612,0.0580582617584078,0.350939923568617,0.941941738241592,1.53945573867507,1.88388347648318,1.98309396859818,1.88388347648318,1.53945573867507,0.941941738241592,0.350939923568617,0.0580582617584078,0.00192289067087612,4.89709696887535e-66,3.58785390555994e-05;3.57253960528771e-67,0.000491808921017412,0.0263582300442202,0.226998098512682,0.773001901487318,1.45624550950204,1.87590433043376,1.99152904502956,1.99152904502956,1.87590433043376,1.45624550950204,0.773001901487318,0.226998098512682,0.0263582300442202,0.000491808921017412,3.57253960528771e-67;3.55729067216782e-129,3.58785390555995e-05,0.00674152351681560,0.103056273456746,0.500000000000000,1.19506387940602,1.77450847648318,1.98309396859818,2,1.98309396859818,1.77450847648318,1.19506387940602,0.500000000000000,0.103056273456746,0.00674152351681560,3.58785390555995e-05;3.57253960528770e-67,3.57253960528771e-67,0.000491808921017412,0.0263582300442202,0.226998098512682,0.773001901487318,1.45624550950204,1.87590433043376,1.99152904502956,1.99152904502956,1.87590433043376,1.45624550950204,0.773001901487318,0.226998098512682,0.0263582300442202,0.000491808921017412;4.89709696887535e-66,3.58785390555994e-05,4.89709696887535e-66,0.00192289067087612,0.0580582617584078,0.350939923568617,0.941941738241592,1.53945573867507,1.88388347648318,1.98309396859818,1.88388347648318,1.53945573867507,0.941941738241592,0.350939923568617,0.0580582617584078,0.00192289067087612;1.91468305543256e-65,0.000491808921017411,0.000491808921017411,1.91468305543257e-65,0.00423547748521932,0.0897582934725954,0.427637966981143,0.995764522514781,1.54600380297464,1.87590433043376,1.87590433043376,1.54600380297464,0.995764522514781,0.427637966981143,0.0897582934725954,0.00423547748521932;4.21739888566864e-65,0.00192289067087611,0.00674152351681560,0.00192289067087611,4.21739888566864e-65,0.00654806429956251,0.109375000000000,0.452073306354487,1,1.53945573867507,1.77450847648318,1.53945573867507,1,0.452073306354487,0.109375000000000,0.00654806429956251;6.52011471590472e-65,0.00423547748521931,0.0263582300442202,0.0263582300442202,0.00423547748521930,6.52011471590472e-65,0.00797914604942122,0.115624714595798,0.453996197025363,0.995764522514781,1.45624550950204,1.45624550950204,0.995764522514781,0.453996197025363,0.115624714595798,0.00797914604942122;7.94508807444975e-65,0.00654806429956252,0.0580582617584078,0.103056273456746,0.0580582617584078,0.00654806429956250,7.94508807444975e-65,0.00843507643138303,0.116116523516816,0.452073306354487,0.941941738241592,1.19506387940602,0.941941738241592,0.452073306354487,0.116116523516816,0.00843507643138303;8.39907237528441e-65,0.00797914604942122,0.0897582934725955,0.226998098512682,0.226998098512681,0.0897582934725953,0.00797914604942120,8.39907237528441e-65,0.00847095497043863,0.115624714595798,0.427637966981143,0.773001901487318,0.773001901487318,0.427637966981143,0.115624714595798,0.00847095497043863;];
                        
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','AmplitudeSpecification');
            subband = 1;
            asActual = step(testCase.sbsp,nPoints,subband,[]);
            
            % Evaluation
            diff = max(abs(asExpctd(:) - asActual(:))./abs(asExpctd(:)));
            testCase.verifyEqual(asActual,asExpctd,'RelTol',1e-14,...
                sprintf('%g',diff));
            
        end

        % Test for vertical positive setting
        function testGetAmplitudeSpecificationK2L2(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            K=2;
            L=2;
            
            % Expected array
            asExpctd = [8.43479777133728e-65,0.00843507643138303,0.109375000000000,0.350939923568617,0.500000000000000,0.350939923568617,0.109375000000000,0.00843507643138303,8.43479777133728e-65,0.00843507643138301,0.109375000000000,0.350939923568616,0.500000000000000,0.350939923568617,0.109375000000000,0.00843507643138303;8.39907237528441e-65,0.00847095497043863,0.115624714595798,0.427637966981143,0.773001901487318,0.773001901487318,0.427637966981143,0.115624714595798,0.00847095497043863,8.39907237528441e-65,0.00797914604942120,0.0897582934725953,0.226998098512681,0.226998098512682,0.0897582934725955,0.00797914604942122;7.94508807444975e-65,0.00843507643138303,0.116116523516816,0.452073306354487,0.941941738241592,1.19506387940602,0.941941738241592,0.452073306354487,0.116116523516816,0.00843507643138303,7.94508807444975e-65,0.00654806429956250,0.0580582617584078,0.103056273456746,0.0580582617584078,0.00654806429956252;6.52011471590472e-65,0.00797914604942122,0.115624714595798,0.453996197025363,0.995764522514781,1.45624550950204,1.45624550950204,0.995764522514781,0.453996197025363,0.115624714595798,0.00797914604942122,6.52011471590472e-65,0.00423547748521930,0.0263582300442202,0.0263582300442202,0.00423547748521931;4.21739888566864e-65,0.00654806429956251,0.109375000000000,0.452073306354487,1,1.53945573867507,1.77450847648318,1.53945573867507,1,0.452073306354487,0.109375000000000,0.00654806429956251,4.21739888566864e-65,0.00192289067087611,0.00674152351681560,0.00192289067087611;1.91468305543257e-65,0.00423547748521932,0.0897582934725954,0.427637966981143,0.995764522514781,1.54600380297464,1.87590433043376,1.87590433043376,1.54600380297464,0.995764522514781,0.427637966981143,0.0897582934725954,0.00423547748521932,1.91468305543257e-65,0.000491808921017411,0.000491808921017411;4.89709696887535e-66,0.00192289067087612,0.0580582617584078,0.350939923568617,0.941941738241592,1.53945573867507,1.88388347648318,1.98309396859818,1.88388347648318,1.53945573867507,0.941941738241592,0.350939923568617,0.0580582617584078,0.00192289067087612,4.89709696887535e-66,3.58785390555994e-05;3.57253960528771e-67,0.000491808921017412,0.0263582300442202,0.226998098512682,0.773001901487318,1.45624550950204,1.87590433043376,1.99152904502956,1.99152904502956,1.87590433043376,1.45624550950204,0.773001901487318,0.226998098512682,0.0263582300442202,0.000491808921017412,3.57253960528771e-67;3.55729067216782e-129,3.58785390555995e-05,0.00674152351681560,0.103056273456746,0.500000000000000,1.19506387940602,1.77450847648318,1.98309396859818,2,1.98309396859818,1.77450847648318,1.19506387940602,0.500000000000000,0.103056273456746,0.00674152351681560,3.58785390555995e-05;3.57253960528770e-67,3.57253960528771e-67,0.000491808921017412,0.0263582300442202,0.226998098512682,0.773001901487318,1.45624550950204,1.87590433043376,1.99152904502956,1.99152904502956,1.87590433043376,1.45624550950204,0.773001901487318,0.226998098512682,0.0263582300442202,0.000491808921017412;4.89709696887535e-66,3.58785390555994e-05,4.89709696887535e-66,0.00192289067087612,0.0580582617584078,0.350939923568617,0.941941738241592,1.53945573867507,1.88388347648318,1.98309396859818,1.88388347648318,1.53945573867507,0.941941738241592,0.350939923568617,0.0580582617584078,0.00192289067087612;1.91468305543256e-65,0.000491808921017411,0.000491808921017411,1.91468305543257e-65,0.00423547748521932,0.0897582934725954,0.427637966981143,0.995764522514781,1.54600380297464,1.87590433043376,1.87590433043376,1.54600380297464,0.995764522514781,0.427637966981143,0.0897582934725954,0.00423547748521932;4.21739888566864e-65,0.00192289067087611,0.00674152351681560,0.00192289067087611,4.21739888566864e-65,0.00654806429956251,0.109375000000000,0.452073306354487,1,1.53945573867507,1.77450847648318,1.53945573867507,1,0.452073306354487,0.109375000000000,0.00654806429956251;6.52011471590472e-65,0.00423547748521931,0.0263582300442202,0.0263582300442202,0.00423547748521930,6.52011471590472e-65,0.00797914604942122,0.115624714595798,0.453996197025363,0.995764522514781,1.45624550950204,1.45624550950204,0.995764522514781,0.453996197025363,0.115624714595798,0.00797914604942122;7.94508807444975e-65,0.00654806429956252,0.0580582617584078,0.103056273456746,0.0580582617584078,0.00654806429956250,7.94508807444975e-65,0.00843507643138303,0.116116523516816,0.452073306354487,0.941941738241592,1.19506387940602,0.941941738241592,0.452073306354487,0.116116523516816,0.00843507643138303;8.39907237528441e-65,0.00797914604942122,0.0897582934725955,0.226998098512682,0.226998098512681,0.0897582934725953,0.00797914604942120,8.39907237528441e-65,0.00847095497043863,0.115624714595798,0.427637966981143,0.773001901487318,0.773001901487318,0.427637966981143,0.115624714595798,0.00847095497043863;];
                        
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','AmplitudeSpecification');
            asActual = step(testCase.sbsp,nPoints,1,[K L]);
            
            % Evaluation
            diff = max(abs(asExpctd(:) - asActual(:))./abs(asExpctd(:)));
            testCase.verifyEqual(asActual,asExpctd,'RelTol',1e-14,...
                sprintf('%g',diff));
            
        end
        
        % Test for vertical positive setting
        function testGetAmplitudeSpecificationK1L1(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 0.0;
            direction = Direction.VERTICAL;
            K=1;
            L=1;
            
            % Expected array
            asExpctd = [2.81159925711243e-65,2.85406039442479e-34,1.09817367547699e-33,2.31456640327424e-33,3.74939945665465e-33,5.18423251003505e-33,6.40062523783230e-33,7.21339287386681e-33,7.49879891330929e-33,7.21339287386681e-33,6.40062523783230e-33,5.18423251003505e-33,3.74939945665465e-33,2.31456640327424e-33,1.09817367547699e-33,2.85406039442479e-34;2.85406039442479e-34,0.00289716278535013,0.0111475843701770,0.0234952128584486,0.0380602337443567,0.0526252546302647,0.0649728831185363,0.0732233047033632,0.0761204674887133,0.0732233047033632,0.0649728831185363,0.0526252546302647,0.0380602337443567,0.0234952128584486,0.0111475843701770,0.00289716278535013;1.09817367547699e-33,0.0111475843701770,0.0428932188134525,0.0904039182607307,0.146446609406726,0.202489300552722,0.250000000000000,0.281745634443276,0.292893218813453,0.281745634443276,0.250000000000000,0.202489300552722,0.146446609406726,0.0904039182607307,0.0428932188134525,0.0111475843701770;2.31456640327424e-33,0.0234952128584486,0.0904039182607307,0.190539872338273,0.308658283817455,0.426776695296637,0.526912649374180,0.593821354776462,0.617316567634910,0.593821354776462,0.526912649374180,0.426776695296637,0.308658283817455,0.190539872338273,0.0904039182607307,0.0234952128584486;3.74939945665465e-33,0.0380602337443567,0.146446609406726,0.308658283817455,0.500000000000000,0.691341716182545,0.853553390593274,0.961939766255644,1.00000000000000,0.961939766255644,0.853553390593274,0.691341716182545,0.500000000000000,0.308658283817455,0.146446609406726,0.0380602337443567;5.18423251003505e-33,0.0526252546302647,0.202489300552722,0.426776695296637,0.691341716182545,0.955906737068453,1.18019413181237,1.33005817773483,1.38268343236509,1.33005817773483,1.18019413181237,0.955906737068453,0.691341716182545,0.426776695296637,0.202489300552722,0.0526252546302647;6.40062523783230e-33,0.0649728831185363,0.250000000000000,0.526912649374180,0.853553390593274,1.18019413181237,1.45710678118655,1.64213389806801,1.70710678118655,1.64213389806801,1.45710678118655,1.18019413181237,0.853553390593274,0.526912649374180,0.250000000000000,0.0649728831185363;7.21339287386681e-33,0.0732233047033632,0.281745634443276,0.593821354776462,0.961939766255644,1.33005817773483,1.64213389806801,1.85065622780792,1.92387953251129,1.85065622780792,1.64213389806801,1.33005817773483,0.961939766255644,0.593821354776462,0.281745634443276,0.0732233047033632;7.49879891330929e-33,0.0761204674887133,0.292893218813453,0.617316567634910,1.00000000000000,1.38268343236509,1.70710678118655,1.92387953251129,2,1.92387953251129,1.70710678118655,1.38268343236509,1.00000000000000,0.617316567634910,0.292893218813453,0.0761204674887133;7.21339287386681e-33,0.0732233047033632,0.281745634443276,0.593821354776462,0.961939766255644,1.33005817773483,1.64213389806801,1.85065622780792,1.92387953251129,1.85065622780792,1.64213389806801,1.33005817773483,0.961939766255644,0.593821354776462,0.281745634443276,0.0732233047033632;6.40062523783230e-33,0.0649728831185363,0.250000000000000,0.526912649374180,0.853553390593274,1.18019413181237,1.45710678118655,1.64213389806801,1.70710678118655,1.64213389806801,1.45710678118655,1.18019413181237,0.853553390593274,0.526912649374180,0.250000000000000,0.0649728831185363;5.18423251003505e-33,0.0526252546302647,0.202489300552722,0.426776695296637,0.691341716182545,0.955906737068453,1.18019413181237,1.33005817773483,1.38268343236509,1.33005817773483,1.18019413181237,0.955906737068453,0.691341716182545,0.426776695296637,0.202489300552722,0.0526252546302647;3.74939945665465e-33,0.0380602337443567,0.146446609406726,0.308658283817455,0.500000000000000,0.691341716182545,0.853553390593274,0.961939766255644,1.00000000000000,0.961939766255644,0.853553390593274,0.691341716182545,0.500000000000000,0.308658283817455,0.146446609406726,0.0380602337443567;2.31456640327424e-33,0.0234952128584486,0.0904039182607307,0.190539872338273,0.308658283817455,0.426776695296637,0.526912649374180,0.593821354776462,0.617316567634910,0.593821354776462,0.526912649374180,0.426776695296637,0.308658283817455,0.190539872338273,0.0904039182607307,0.0234952128584486;1.09817367547699e-33,0.0111475843701770,0.0428932188134525,0.0904039182607307,0.146446609406726,0.202489300552722,0.250000000000000,0.281745634443276,0.292893218813453,0.281745634443276,0.250000000000000,0.202489300552722,0.146446609406726,0.0904039182607307,0.0428932188134525,0.0111475843701770;2.85406039442479e-34,0.00289716278535013,0.0111475843701770,0.0234952128584486,0.0380602337443567,0.0526252546302647,0.0649728831185363,0.0732233047033632,0.0761204674887133,0.0732233047033632,0.0649728831185363,0.0526252546302647,0.0380602337443567,0.0234952128584486,0.0111475843701770,0.00289716278535013;];
                        
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','AmplitudeSpecification');
            asActual = step(testCase.sbsp,nPoints,1,[K L]);
            
            % Evaluation
            diff = max(abs(asExpctd(:) - asActual(:))./abs(asExpctd(:)));
            testCase.verifyEqual(asActual,asExpctd,'RelTol',1e-14,...
                sprintf('%g',diff));
            
        end

        % Test for vertical positive setting
        function testGetAmplitudeSpecificationK3L1(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = -2.0;
            direction = Direction.VERTICAL;
            K=3;
            L=1;
            
            % Expected array
            asExpctd = [4.05067870353601e-192,3.46323779689901e-07,0.000785195807721366,0.0365726039141742,0.250000000000000,0.410960765206622,0.155464804192279,0.00559128455542420,1.05418087269479e-97,0.00559128455542421,0.155464804192279,0.410960765206622,0.250000000000000,0.0365726039141742,0.000785195807721366,3.46323779689898e-07;5.81205192256348e-102,3.24248637968853e-06,0.00207561236547861,0.0523488858667440,0.222527477399638,0.218366782973290,0.0365726039141740,9.81494759651703e-05,0.000110266692805877,0.0523488858667440,0.410960765206622,0.588236436322586,0.222527477399638,0.0194331005240348,0.000184714832531142,6.07937177117285e-09;3.31095360728004e-100,1.37833366007347e-05,0.00390625000000001,0.0588117099947285,0.155464804192279,0.0826072755144201,0.00390625000000000,9.38336841294886e-98,0.00628156646177092,0.222527477399638,0.773417770961393,0.660858204115361,0.155464804192279,0.00735146374934106,1.97290386068226e-05,5.81205192256342e-102;3.09990898834580e-99,3.64353242907162e-05,0.00559128455542421,0.0523488858667440,0.0826072755144202,0.0194331005240348,6.85703592239831e-05,9.81494759651707e-05,0.0588117099947285,0.588236436322586,1.10704610244277,0.588236436322586,0.0826072755144202,0.00172940861625202,3.46323779689898e-07,6.07937177117283e-09;1.31772609086849e-98,6.85703592239834e-05,0.00628156646177092,0.0365726039141742,0.0312500000000000,0.00207561236547860,6.55552091826966e-98,0.00559128455542421,0.250000000000000,1.10704610244277,1.24371843353823,0.410960765206622,0.0312500000000000,0.000184714832531142,3.31095360728001e-100,3.46323779689897e-07;3.48332039170923e-98,9.81494759651707e-05,0.00559128455542421,0.0194331005240349,0.00735146374934104,3.64353242907160e-05,6.85703592239834e-05,0.0523488858667440,0.660858204115361,1.58459130233109,1.10704610244277,0.218366782973290,0.00735146374934106,3.24248637968853e-06,3.46323779689896e-07,3.24248637968852e-06;6.55552091826967e-98,0.000110266692805877,0.00390625000000000,0.00735146374934106,0.000785195807721364,3.48332039170923e-98,0.00390625000000000,0.222527477399638,1.24371843353823,1.78021981919710,0.773417770961393,0.0826072755144202,0.000785195807721366,3.09990898834580e-99,1.97290386068226e-05,1.37833366007346e-05;9.38336841294886e-98,9.81494759651707e-05,0.00207561236547861,0.00172940861625202,1.37833366007346e-05,3.64353242907161e-05,0.0365726039141742,0.588236436322586,1.78021981919710,1.58459130233109,0.410960765206622,0.0194331005240348,1.37833366007347e-05,3.24248637968852e-06,0.000184714832531141,3.64353242907162e-05;1.05418087269479e-97,6.85703592239834e-05,0.000785195807721365,0.000184714832531142,1.31772609086849e-98,0.00207561236547861,0.155464804192279,1.10704610244277,2,1.10704610244277,0.155464804192279,0.00207561236547861,1.31772609086849e-98,0.000184714832531142,0.000785195807721365,6.85703592239834e-05;9.38336841294886e-98,3.64353242907162e-05,0.000184714832531141,3.24248637968852e-06,1.37833366007347e-05,0.0194331005240348,0.410960765206622,1.58459130233109,1.78021981919710,0.588236436322586,0.0365726039141742,3.64353242907161e-05,1.37833366007346e-05,0.00172940861625202,0.00207561236547861,9.81494759651707e-05;6.55552091826966e-98,1.37833366007346e-05,1.97290386068226e-05,3.09990898834580e-99,0.000785195807721366,0.0826072755144202,0.773417770961393,1.78021981919710,1.24371843353823,0.222527477399638,0.00390625000000000,3.48332039170923e-98,0.000785195807721364,0.00735146374934106,0.00390625000000000,0.000110266692805877;3.48332039170923e-98,3.24248637968852e-06,3.46323779689896e-07,3.24248637968853e-06,0.00735146374934106,0.218366782973290,1.10704610244277,1.58459130233109,0.660858204115361,0.0523488858667440,6.85703592239834e-05,3.64353242907160e-05,0.00735146374934104,0.0194331005240349,0.00559128455542421,9.81494759651707e-05;1.31772609086849e-98,3.46323779689897e-07,3.31095360728001e-100,0.000184714832531142,0.0312500000000000,0.410960765206622,1.24371843353823,1.10704610244277,0.250000000000000,0.00559128455542421,6.55552091826966e-98,0.00207561236547860,0.0312500000000000,0.0365726039141742,0.00628156646177092,6.85703592239834e-05;3.09990898834579e-99,6.07937177117283e-09,3.46323779689898e-07,0.00172940861625202,0.0826072755144202,0.588236436322586,1.10704610244277,0.588236436322586,0.0588117099947285,9.81494759651707e-05,6.85703592239831e-05,0.0194331005240348,0.0826072755144202,0.0523488858667440,0.00559128455542421,3.64353242907162e-05;3.31095360728000e-100,5.81205192256342e-102,1.97290386068226e-05,0.00735146374934106,0.155464804192279,0.660858204115361,0.773417770961393,0.222527477399638,0.00628156646177092,9.38336841294886e-98,0.00390625000000000,0.0826072755144201,0.155464804192279,0.0588117099947285,0.00390625000000001,1.37833366007347e-05;5.81205192256340e-102,6.07937177117285e-09,0.000184714832531142,0.0194331005240348,0.222527477399638,0.588236436322586,0.410960765206622,0.0523488858667440,0.000110266692805877,9.81494759651703e-05,0.0365726039141740,0.218366782973290,0.222527477399638,0.0523488858667440,0.00207561236547861,3.24248637968853e-06;];
                        
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','AmplitudeSpecification');
            asActual = step(testCase.sbsp,nPoints,1,[K L]);
            
            % Evaluation
            diff = max(abs(asExpctd(:) - asActual(:))./abs(asExpctd(:)));
            testCase.verifyEqual(asActual,asExpctd,'RelTol',1e-14,...
                sprintf('%g',diff));
            
        end
               
        % Test for vertical positive setting
        function testGetAmplitudeSpecificationHyLx(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            
            % Expected array
            asExpctd = [3.55729067216782e-129,3.58785390555995e-05,0.00674152351681560,0.103056273456746,0.500000000000000,1.19506387940602,1.77450847648318,1.98309396859818,2,1.98309396859818,1.77450847648318,1.19506387940602,0.500000000000000,0.103056273456746,0.00674152351681560,3.58785390555995e-05;3.57253960528770e-67,3.57253960528771e-67,0.000491808921017412,0.0263582300442202,0.226998098512682,0.773001901487318,1.45624550950204,1.87590433043376,1.99152904502956,1.99152904502956,1.87590433043376,1.45624550950204,0.773001901487318,0.226998098512682,0.0263582300442202,0.000491808921017412;4.89709696887535e-66,3.58785390555994e-05,4.89709696887535e-66,0.00192289067087612,0.0580582617584078,0.350939923568617,0.941941738241592,1.53945573867507,1.88388347648318,1.98309396859818,1.88388347648318,1.53945573867507,0.941941738241592,0.350939923568617,0.0580582617584078,0.00192289067087612;1.91468305543256e-65,0.000491808921017411,0.000491808921017411,1.91468305543257e-65,0.00423547748521932,0.0897582934725954,0.427637966981143,0.995764522514781,1.54600380297464,1.87590433043376,1.87590433043376,1.54600380297464,0.995764522514781,0.427637966981143,0.0897582934725954,0.00423547748521932;4.21739888566864e-65,0.00192289067087611,0.00674152351681560,0.00192289067087611,4.21739888566864e-65,0.00654806429956251,0.109375000000000,0.452073306354487,1,1.53945573867507,1.77450847648318,1.53945573867507,1,0.452073306354487,0.109375000000000,0.00654806429956251;6.52011471590472e-65,0.00423547748521931,0.0263582300442202,0.0263582300442202,0.00423547748521930,6.52011471590472e-65,0.00797914604942122,0.115624714595798,0.453996197025363,0.995764522514781,1.45624550950204,1.45624550950204,0.995764522514781,0.453996197025363,0.115624714595798,0.00797914604942122;7.94508807444975e-65,0.00654806429956252,0.0580582617584078,0.103056273456746,0.0580582617584078,0.00654806429956250,7.94508807444975e-65,0.00843507643138303,0.116116523516816,0.452073306354487,0.941941738241592,1.19506387940602,0.941941738241592,0.452073306354487,0.116116523516816,0.00843507643138303;8.39907237528441e-65,0.00797914604942122,0.0897582934725955,0.226998098512682,0.226998098512681,0.0897582934725953,0.00797914604942120,8.39907237528441e-65,0.00847095497043863,0.115624714595798,0.427637966981143,0.773001901487318,0.773001901487318,0.427637966981143,0.115624714595798,0.00847095497043863;8.43479777133728e-65,0.00843507643138303,0.109375000000000,0.350939923568617,0.500000000000000,0.350939923568616,0.109375000000000,0.00843507643138301,8.43479777133728e-65,0.00843507643138303,0.109375000000000,0.350939923568617,0.500000000000000,0.350939923568617,0.109375000000000,0.00843507643138303;8.39907237528441e-65,0.00847095497043863,0.115624714595798,0.427637966981143,0.773001901487319,0.773001901487318,0.427637966981142,0.115624714595798,0.00847095497043861,8.39907237528441e-65,0.00797914604942122,0.0897582934725954,0.226998098512682,0.226998098512682,0.0897582934725954,0.00797914604942122;7.94508807444975e-65,0.00843507643138303,0.116116523516816,0.452073306354487,0.941941738241592,1.19506387940602,0.941941738241592,0.452073306354486,0.116116523516816,0.00843507643138301,7.94508807444975e-65,0.00654806429956251,0.0580582617584078,0.103056273456746,0.0580582617584078,0.00654806429956251;6.52011471590472e-65,0.00797914604942122,0.115624714595798,0.453996197025363,0.995764522514781,1.45624550950204,1.45624550950204,0.995764522514780,0.453996197025362,0.115624714595798,0.00797914604942116,2.98659774188803e-62,0.00423547748521933,0.0263582300442203,0.0263582300442203,0.00423547748521932;4.21739888566864e-65,0.00654806429956252,0.109375000000000,0.452073306354487,1,1.53945573867507,1.77450847648318,1.53945573867508,1.00000000000000,0.452073306354486,0.109375000000000,0.00654806429956250,4.21739888566864e-65,0.00192289067087612,0.00674152351681560,0.00192289067087612;1.91468305543257e-65,0.00423547748521932,0.0897582934725955,0.427637966981143,0.995764522514781,1.54600380297464,1.87590433043376,1.87590433043376,1.54600380297464,0.995764522514780,0.427637966981143,0.0897582934725955,0.00423547748521932,9.10856883954793e-64,0.000491808921017410,0.000491808921017411;4.89709696887538e-66,0.00192289067087612,0.0580582617584079,0.350939923568617,0.941941738241592,1.53945573867507,1.88388347648318,1.98309396859818,1.88388347648318,1.53945573867508,0.941941738241592,0.350939923568616,0.0580582617584078,0.00192289067087611,4.89709696887535e-66,3.58785390555995e-05;3.57253960528774e-67,0.000491808921017415,0.0263582300442202,0.226998098512682,0.773001901487319,1.45624550950204,1.87590433043376,1.99152904502956,1.99152904502956,1.87590433043376,1.45624550950204,0.773001901487318,0.226998098512681,0.0263582300442202,0.000491808921017408,1.63643420136931e-64;];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','AmplitudeSpecification');
            asActual = step(testCase.sbsp,nPoints,2,[]);
            
            % Evaluation
            diff = max(abs(asExpctd(:) - asActual(:))./abs(asExpctd(:)));
            testCase.verifyEqual(asActual,asExpctd,'RelTol',1e-14,...
                sprintf('%g',diff));

        end
        
        % Test for vertical positive setting
        function testGetAmplitudeSpecificationLyHx(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = -2.0;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            asExpctd = [8.43479777133728e-65,8.39907237528441e-65,7.94508807444975e-65,6.52011471590472e-65,4.21739888566864e-65,1.91468305543256e-65,4.89709696887535e-66,3.57253960528770e-67,3.55729067216782e-129,3.57253960528771e-67,4.89709696887535e-66,1.91468305543257e-65,4.21739888566864e-65,6.52011471590472e-65,7.94508807444975e-65,8.39907237528441e-65;0.00797914604942122,0.00654806429956252,0.00423547748521931,0.00192289067087611,0.000491808921017411,3.58785390555994e-05,3.57253960528771e-67,3.58785390555995e-05,0.000491808921017412,0.00192289067087612,0.00423547748521932,0.00654806429956251,0.00797914604942122,0.00843507643138303,0.00847095497043863,0.00843507643138303;0.0580582617584078,0.0263582300442202,0.00674152351681560,0.000491808921017411,4.89709696887535e-66,0.000491808921017412,0.00674152351681560,0.0263582300442202,0.0580582617584078,0.0897582934725954,0.109375000000000,0.115624714595798,0.116116523516816,0.115624714595798,0.109375000000000,0.0897582934725955;0.0263582300442202,0.00192289067087611,1.91468305543257e-65,0.00192289067087612,0.0263582300442202,0.103056273456746,0.226998098512682,0.350939923568617,0.427637966981143,0.452073306354487,0.453996197025363,0.452073306354487,0.427637966981143,0.350939923568617,0.226998098512682,0.103056273456746;4.21739888566864e-65,0.00423547748521932,0.0580582617584078,0.226998098512682,0.500000000000000,0.773001901487318,0.941941738241592,0.995764522514781,1,0.995764522514781,0.941941738241592,0.773001901487319,0.500000000000000,0.226998098512681,0.0580582617584078,0.00423547748521933;0.0897582934725954,0.350939923568617,0.773001901487318,1.19506387940602,1.45624550950204,1.53945573867507,1.54600380297464,1.53945573867507,1.45624550950204,1.19506387940602,0.773001901487318,0.350939923568617,0.0897582934725954,0.00654806429956248,6.52011471590472e-65,0.00654806429956247;0.941941738241592,1.45624550950204,1.77450847648318,1.87590433043376,1.88388347648318,1.87590433043376,1.77450847648318,1.45624550950204,0.941941738241592,0.427637966981143,0.109375000000000,0.00797914604942125,7.94508807444975e-65,0.00797914604942123,0.109375000000000,0.427637966981142;1.87590433043376,1.98309396859818,1.99152904502956,1.98309396859818,1.87590433043376,1.53945573867507,0.995764522514781,0.452073306354487,0.115624714595798,0.00843507643138303,8.39907237528441e-65,0.00843507643138297,0.115624714595798,0.452073306354487,0.995764522514780,1.53945573867507;2,1.99152904502956,1.88388347648318,1.54600380297464,1,0.453996197025363,0.116116523516816,0.00847095497043863,8.43479777133728e-65,0.00847095497043861,0.116116523516816,0.453996197025362,1.00000000000000,1.54600380297464,1.88388347648318,1.99152904502956;1.87590433043376,1.53945573867507,0.995764522514781,0.452073306354487,0.115624714595798,0.00843507643138303,8.39907237528441e-65,0.00843507643138301,0.115624714595798,0.452073306354486,0.995764522514780,1.53945573867507,1.87590433043376,1.98309396859818,1.99152904502956,1.98309396859818;0.941941738241592,0.427637966981143,0.109375000000000,0.00797914604942122,7.94508807444975e-65,0.00797914604942120,0.109375000000000,0.427637966981142,0.941941738241592,1.45624550950204,1.77450847648318,1.87590433043376,1.88388347648318,1.87590433043376,1.77450847648318,1.45624550950204;0.0897582934725954,0.00654806429956251,6.52011471590472e-65,0.00654806429956250,0.0897582934725953,0.350939923568616,0.773001901487318,1.19506387940602,1.45624550950204,1.53945573867507,1.54600380297464,1.53945573867507,1.45624550950204,1.19506387940602,0.773001901487319,0.350939923568617;4.21739888566864e-65,0.00423547748521930,0.0580582617584078,0.226998098512681,0.500000000000000,0.773001901487319,0.941941738241592,0.995764522514781,1,0.995764522514781,0.941941738241592,0.773001901487319,0.500000000000000,0.226998098512682,0.0580582617584081,0.00423547748521934;0.0263582300442202,0.103056273456746,0.226998098512682,0.350939923568617,0.427637966981143,0.452073306354487,0.453996197025363,0.452073306354487,0.427637966981143,0.350939923568617,0.226998098512682,0.103056273456746,0.0263582300442204,0.00192289067087610,1.55089327490038e-63,0.00192289067087608;0.0580582617584078,0.0897582934725955,0.109375000000000,0.115624714595798,0.116116523516816,0.115624714595798,0.109375000000000,0.0897582934725955,0.0580582617584079,0.0263582300442202,0.00674152351681564,0.000491808921017415,3.96664854478903e-64,0.000491808921017412,0.00674152351681561,0.0263582300442202;0.00797914604942122,0.00843507643138303,0.00847095497043863,0.00843507643138303,0.00797914604942122,0.00654806429956252,0.00423547748521932,0.00192289067087612,0.000491808921017415,3.58785390555997e-05,2.89375708028305e-65,3.58785390555988e-05,0.000491808921017413,0.00192289067087611,0.00423547748521931,0.00654806429956251;];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','AmplitudeSpecification');
            asActual = step(testCase.sbsp,nPoints,3,[]);
            
            % Evaluation
            diff = max(abs(asExpctd(:) - asActual(:))./abs(asExpctd(:)));
            testCase.verifyEqual(asActual,asExpctd,'RelTol',1e-14,...
                sprintf('%g',diff));
            
        end
        
        % Test for vertical positive setting
        function testGetAmplitudeSpecificationHyHx(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [32 32];
            alpha = 1.0;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            asExpctd = [2,1.99944973891582,1.99152904502956,1.95978960409238,1.88388347648318,1.74761467219746,1.54600380297464,1.28892289139559,1,0.711077108604405,0.453996197025363,0.252385327802544,0.116116523516816,0.0402103959076206,0.00847095497043863,0.000550261084183903,8.43479777133728e-65,0.000550261084183895,0.00847095497043857,0.0402103959076205,0.116116523516816,0.252385327802544,0.453996197025362,0.711077108604405,1.00000000000000,1.28892289139559,1.54600380297464,1.74761467219746,1.88388347648318,1.95978960409238,1.99152904502956,1.99944973891582;1.99889962922526,1.99944973891582,1.99889962922526,1.99098111456381,1.95925040611622,1.88336516260106,1.74713385002533,1.54557845011025,1.28856826934177,0.999724869457909,0.710881469574046,0.453871288805569,0.252315888890490,0.116084576314755,0.0401993327995969,0.00846862435200561,0.000550109690553520,3.86257269469509e-62,0.000550109690553507,0.00846862435200552,0.0401993327995967,0.116084576314755,0.252315888890489,0.453871288805567,0.710881469574046,0.999724869457906,1.28856826934177,1.54557845011025,1.74713385002533,1.88336516260106,1.95925040611622,1.99098111456381;1.98309396859818,1.99098111456381,1.99152904502956,1.99098111456381,1.98309396859818,1.95148895934848,1.87590433043376,1.74021268960053,1.53945573867507,1.28346368750890,0.995764522514781,0.708065357520657,0.452073306354487,0.251316355429036,0.115624714595798,0.0400400856810822,0.00843507643138303,0.000547930465750850,3.84727135686567e-62,0.000547930465750842,0.00843507643138301,0.0400400856810821,0.115624714595798,0.251316355429036,0.452073306354486,0.708065357520655,0.995764522514780,1.28346368750890,1.53945573867508,1.74021268960052,1.87590433043376,1.95148895934848;1.92038764615428,1.95148895934848,1.95925040611622,1.95978960409238,1.95925040611622,1.95148895934848,1.92038764615428,1.84600762626658,1.71247853326594,1.51492109047849,1.26300884151689,0.979894802046190,0.696780762575491,0.444868513613891,0.247311070826436,0.113781977825802,0.0394019579380968,0.00830064474390015,0.000539197976160108,8.26521449244419e-65,0.000539197976160105,0.00830064474390017,0.0394019579380967,0.113781977825802,0.247311070826436,0.444868513613890,0.696780762575491,0.979894802046190,1.26300884151689,1.51492109047849,1.71247853326594,1.84600762626658;1.77450847648318,1.84600762626658,1.87590433043376,1.88336516260106,1.88388347648318,1.88336516260106,1.87590433043376,1.84600762626658,1.77450847648318,1.64615120210618,1.45624550950204,1.21409026878055,0.941941738241592,0.669793207702639,0.427637966981143,0.237732274377002,0.109375000000000,0.0378758502166068,0.00797914604942125,0.000518313882122889,7.94508807444975e-65,0.000518313882122886,0.00797914604942116,0.0378758502166066,0.109375000000000,0.237732274377001,0.427637966981142,0.669793207702639,0.941941738241592,1.21409026878055,1.45624550950204,1.64615120210618;1.52707852123991,1.64615120210618,1.71247853326594,1.74021268960053,1.74713385002533,1.74761467219746,1.74713385002533,1.74021268960052,1.71247853326594,1.64615120210618,1.52707852123991,1.35090946467577,1.12627027816706,0.873807336098728,0.621344394030402,0.396705207521686,0.220536150957545,0.101463470091274,0.0351361389315133,0.00740198259693129,0.000480822172129534,7.37038817110372e-65,0.000480822172129523,0.00740198259693121,0.0351361389315130,0.101463470091273,0.220536150957544,0.396705207521685,0.621344394030401,0.873807336098728,1.12627027816705,1.35090946467577;1.19506387940602,1.35090946467577,1.45624550950204,1.51492109047849,1.53945573867507,1.54557845011025,1.54600380297464,1.54557845011025,1.53945573867507,1.51492109047849,1.45624550950204,1.35090946467577,1.19506387940602,0.996339845919327,0.773001901487318,0.549663957055310,0.350939923568617,0.195094338298867,0.0897582934725956,0.0310827124961486,0.00654806429956251,0.000425352864388627,2.98659774188803e-62,0.000425352864388624,0.00654806429956250,0.0310827124961484,0.0897582934725952,0.195094338298867,0.350939923568616,0.549663957055311,0.773001901487318,0.996339845919326;0.830661109981790,0.996339845919327,1.12627027816705,1.21409026878055,1.26300884151689,1.28346368750890,1.28856826934177,1.28892289139559,1.28856826934177,1.28346368750890,1.26300884151689,1.21409026878055,1.12627027816705,0.996339845919327,0.830661109981790,0.644461445697798,0.458261781413805,0.292583045476268,0.162652613228540,0.0748326226150493,0.0259140498787059,0.00545920388668980,0.000354622053824396,5.43590196588458e-65,0.000354622053824394,0.00545920388668978,0.0259140498787059,0.0748326226150492,0.162652613228540,0.292583045476268,0.458261781413805,0.644461445697797;0.500000000000000,0.644461445697798,0.773001901487318,0.873807336098728,0.941941738241592,0.979894802046190,0.995764522514781,0.999724869457908,1,0.999724869457908,0.995764522514781,0.979894802046190,0.941941738241592,0.873807336098728,0.773001901487318,0.644461445697798,0.500000000000000,0.355538554302203,0.226998098512682,0.126192663901272,0.0580582617584078,0.0201051979538103,0.00423547748521933,0.000275130542091952,4.21739888566864e-65,0.000275130542091945,0.00423547748521928,0.0201051979538102,0.0580582617584078,0.126192663901272,0.226998098512681,0.355538554302203;0.252815327190601,0.355538554302203,0.458261781413805,0.549663957055310,0.621344394030401,0.669793207702639,0.696780762575490,0.708065357520656,0.710881469574046,0.711077108604405,0.710881469574046,0.708065357520656,0.696780762575490,0.669793207702639,0.621344394030401,0.549663957055310,0.458261781413805,0.355538554302203,0.252815327190601,0.161413151549096,0.0897327145740040,0.0412839009017664,0.0142963460289147,0.00301175108374882,0.000195639030359508,1.39094267029665e-61,0.000195639030359503,0.00301175108374879,0.0142963460289146,0.0412839009017663,0.0897327145740036,0.161413151549095;0.103056273456746,0.161413151549096,0.226998098512682,0.292583045476268,0.350939923568617,0.396705207521686,0.427637966981143,0.444868513613891,0.452073306354487,0.453871288805568,0.453996197025363,0.453871288805568,0.452073306354487,0.444868513613891,0.427637966981143,0.396705207521686,0.350939923568617,0.292583045476268,0.226998098512682,0.161413151549096,0.103056273456746,0.0572909895036772,0.0263582300442203,0.00912768341147200,0.00192289067087612,0.000124908219795274,8.77037956991332e-63,0.000124908219795271,0.00192289067087611,0.00912768341147199,0.0263582300442202,0.0572909895036773;0.0318491768449988,0.0572909895036774,0.0897327145740041,0.126192663901272,0.162652613228540,0.195094338298867,0.220536150957545,0.237732274377002,0.247311070826437,0.251316355429037,0.252315888890490,0.252385327802544,0.252315888890490,0.251316355429037,0.247311070826437,0.237732274377002,0.220536150957545,0.195094338298867,0.162652613228540,0.126192663901272,0.0897327145740040,0.0572909895036773,0.0318491768449988,0.0146530534255417,0.00507425697610745,0.00106897237350738,6.94389120543688e-05,1.06440960023356e-65,6.94389120543685e-05,0.00106897237350737,0.00507425697610743,0.0146530534255417;0.00674152351681560,0.0146530534255417,0.0263582300442202,0.0412839009017664,0.0580582617584078,0.0748326226150494,0.0897582934725954,0.101463470091274,0.109375000000000,0.113781977825802,0.115624714595798,0.116084576314755,0.116116523516816,0.116084576314755,0.115624714595798,0.113781977825802,0.109375000000000,0.101463470091274,0.0897582934725955,0.0748326226150494,0.0580582617584078,0.0412839009017664,0.0263582300442203,0.0146530534255417,0.00674152351681560,0.00233454569101386,0.000491808921017414,3.19472020610143e-05,4.89709696887535e-66,3.19472020610142e-05,0.000491808921017408,0.00233454569101384;0.000808437969523794,0.00233454569101385,0.00507425697610744,0.00912768341147197,0.0142963460289146,0.0201051979538103,0.0259140498787059,0.0310827124961486,0.0351361389315131,0.0378758502166067,0.0394019579380968,0.0400400856810821,0.0401993327995968,0.0402103959076206,0.0401993327995968,0.0400400856810821,0.0394019579380968,0.0378758502166067,0.0351361389315131,0.0310827124961486,0.0259140498787060,0.0201051979538103,0.0142963460289147,0.00912768341147199,0.00507425697610745,0.00233454569101385,0.000808437969523798,0.000170310226538483,1.10631080237956e-05,1.69583278893094e-66,1.10631080237954e-05,0.000170310226538481;3.58785390555995e-05,0.000170310226538482,0.000491808921017412,0.00106897237350737,0.00192289067087612,0.00301175108374881,0.00423547748521932,0.00545920388668982,0.00654806429956251,0.00740198259693126,0.00797914604942122,0.00830064474390015,0.00843507643138303,0.00846862435200558,0.00847095497043863,0.00846862435200558,0.00843507643138303,0.00830064474390015,0.00797914604942122,0.00740198259693126,0.00654806429956251,0.00545920388668982,0.00423547748521932,0.00301175108374881,0.00192289067087612,0.00106897237350737,0.000491808921017413,0.000170310226538482,3.58785390555995e-05,2.33061843305327e-06,1.63643420136931e-64,2.33061843305326e-06;1.51393630383622e-07,2.33061843305330e-06,1.10631080237957e-05,3.19472020610143e-05,6.94389120543688e-05,0.000124908219795273,0.000195639030359508,0.000275130542091952,0.000354622053824396,0.000425352864388631,0.000480822172129534,0.000518313882122889,0.000539197976160108,0.000547930465750850,0.000550109690553520,0.000550261084183903,0.000550109690553520,0.000547930465750850,0.000539197976160108,0.000518313882122889,0.000480822172129534,0.000425352864388630,0.000354622053824396,0.000275130542091952,0.000195639030359508,0.000124908219795273,6.94389120543688e-05,3.19472020610143e-05,1.10631080237956e-05,2.33061843305328e-06,1.51393630383622e-07,2.32067048326401e-68;3.55729067216782e-129,2.32067048326401e-68,3.57253960528771e-67,1.69583278893094e-66,4.89709696887535e-66,1.06440960023356e-65,1.91468305543257e-65,2.99889580545270e-65,4.21739888566864e-65,5.43590196588458e-65,6.52011471590472e-65,7.37038817110372e-65,7.94508807444975e-65,8.26521449244419e-65,8.39907237528441e-65,8.43247710085402e-65,8.43479777133728e-65,8.43247710085402e-65,8.39907237528441e-65,8.26521449244419e-65,7.94508807444975e-65,7.37038817110372e-65,6.52011471590472e-65,5.43590196588459e-65,4.21739888566864e-65,2.99889580545271e-65,1.91468305543257e-65,1.06440960023356e-65,4.89709696887535e-66,1.69583278893094e-66,3.57253960528773e-67,2.32067048326401e-68;1.51393630383618e-07,1.06300418427848e-65,1.51393630383620e-07,2.33061843305326e-06,1.10631080237955e-05,3.19472020610139e-05,6.94389120543678e-05,0.000124908219795271,0.000195639030359505,0.000275130542091948,0.000354622053824391,0.000425352864388624,0.000480822172129528,0.000518313882122882,0.000539197976160100,0.000547930465750842,0.000550109690553512,0.000550261084183895,0.000550109690553512,0.000547930465750842,0.000539197976160100,0.000518313882122881,0.000480822172129528,0.000425352864388624,0.000354622053824390,0.000275130542091948,0.000195639030359505,0.000124908219795271,6.94389120543677e-05,3.19472020610138e-05,1.10631080237955e-05,2.33061843305326e-06;3.58785390555990e-05,2.33061843305322e-06,1.63643420136930e-64,2.33061843305328e-06,3.58785390555994e-05,0.000170310226538481,0.000491808921017409,0.00106897237350736,0.00192289067087611,0.00301175108374879,0.00423547748521929,0.00545920388668979,0.00654806429956247,0.00740198259693121,0.00797914604942116,0.00830064474390009,0.00843507643138297,0.00846862435200552,0.00847095497043857,0.00846862435200552,0.00843507643138297,0.00830064474390009,0.00797914604942116,0.00740198259693121,0.00654806429956246,0.00545920388668979,0.00423547748521928,0.00301175108374879,0.00192289067087610,0.00106897237350736,0.000491808921017408,0.000170310226538481;0.000808437969523790,0.000170310226538480,1.10631080237955e-05,1.69583278893094e-66,1.10631080237956e-05,0.000170310226538482,0.000808437969523794,0.00233454569101384,0.00507425697610743,0.00912768341147197,0.0142963460289146,0.0201051979538102,0.0259140498787059,0.0310827124961485,0.0351361389315130,0.0378758502166066,0.0394019579380967,0.0400400856810820,0.0401993327995967,0.0402103959076205,0.0401993327995967,0.0400400856810820,0.0394019579380967,0.0378758502166066,0.0351361389315130,0.0310827124961485,0.0259140498787059,0.0201051979538102,0.0142963460289146,0.00912768341147195,0.00507425697610743,0.00233454569101384;0.00674152351681559,0.00233454569101384,0.000491808921017410,3.19472020610141e-05,4.89709696887535e-66,3.19472020610143e-05,0.000491808921017411,0.00233454569101384,0.00674152351681560,0.0146530534255417,0.0263582300442202,0.0412839009017663,0.0580582617584078,0.0748326226150493,0.0897582934725953,0.101463470091274,0.109375000000000,0.113781977825802,0.115624714595798,0.116084576314755,0.116116523516816,0.116084576314755,0.115624714595798,0.113781977825802,0.109375000000000,0.101463470091274,0.0897582934725954,0.0748326226150493,0.0580582617584078,0.0412839009017663,0.0263582300442202,0.0146530534255417;0.0318491768449987,0.0146530534255416,0.00507425697610744,0.00106897237350737,6.94389120543684e-05,1.06440960023356e-65,6.94389120543681e-05,0.00106897237350737,0.00507425697610744,0.0146530534255417,0.0318491768449987,0.0572909895036772,0.0897327145740040,0.126192663901272,0.162652613228540,0.195094338298866,0.220536150957545,0.237732274377002,0.247311070826436,0.251316355429036,0.252315888890489,0.252385327802544,0.252315888890489,0.251316355429036,0.247311070826436,0.237732274377002,0.220536150957545,0.195094338298867,0.162652613228540,0.126192663901272,0.0897327145740041,0.0572909895036773;0.103056273456746,0.0572909895036770,0.0263582300442201,0.00912768341147194,0.00192289067087610,0.000124908219795269,8.77037956991330e-63,0.000124908219795272,0.00192289067087612,0.00912768341147201,0.0263582300442202,0.0572909895036772,0.103056273456746,0.161413151549096,0.226998098512681,0.292583045476267,0.350939923568617,0.396705207521685,0.427637966981142,0.444868513613890,0.452073306354486,0.453871288805567,0.453996197025362,0.453871288805567,0.452073306354486,0.444868513613890,0.427637966981142,0.396705207521685,0.350939923568616,0.292583045476267,0.226998098512681,0.161413151549095;0.252815327190600,0.161413151549095,0.0897327145740039,0.0412839009017663,0.0142963460289146,0.00301175108374879,0.000195639030359505,2.99889580545270e-65,0.000195639030359508,0.00301175108374882,0.0142963460289146,0.0412839009017664,0.0897327145740040,0.161413151549096,0.252815327190601,0.355538554302203,0.458261781413805,0.549663957055310,0.621344394030401,0.669793207702639,0.696780762575491,0.708065357520657,0.710881469574046,0.711077108604405,0.710881469574046,0.708065357520657,0.696780762575491,0.669793207702639,0.621344394030401,0.549663957055309,0.458261781413805,0.355538554302203;0.500000000000000,0.355538554302202,0.226998098512681,0.126192663901272,0.0580582617584077,0.0201051979538102,0.00423547748521930,0.000275130542091950,4.21739888566864e-65,0.000275130542091951,0.00423547748521931,0.0201051979538103,0.0580582617584078,0.126192663901272,0.226998098512682,0.355538554302203,0.500000000000000,0.644461445697797,0.773001901487318,0.873807336098728,0.941941738241592,0.979894802046190,0.995764522514781,0.999724869457908,1.00000000000000,0.999724869457908,0.995764522514781,0.979894802046190,0.941941738241592,0.873807336098728,0.773001901487319,0.644461445697798;0.830661109981788,0.644461445697795,0.458261781413803,0.292583045476267,0.162652613228539,0.0748326226150488,0.0259140498787058,0.00545920388668978,0.000354622053824387,2.52127065640313e-61,0.000354622053824398,0.00545920388668984,0.0259140498787060,0.0748326226150495,0.162652613228540,0.292583045476268,0.458261781413805,0.644461445697797,0.830661109981789,0.996339845919326,1.12627027816705,1.21409026878054,1.26300884151689,1.28346368750890,1.28856826934177,1.28892289139559,1.28856826934177,1.28346368750890,1.26300884151689,1.21409026878054,1.12627027816705,0.996339845919326;1.19506387940602,0.996339845919325,0.773001901487318,0.549663957055309,0.350939923568616,0.195094338298866,0.0897582934725951,0.0310827124961485,0.00654806429956246,0.000425352864388620,2.98659774188803e-62,0.000425352864388630,0.00654806429956254,0.0310827124961487,0.0897582934725955,0.195094338298867,0.350939923568617,0.549663957055310,0.773001901487318,0.996339845919327,1.19506387940602,1.35090946467577,1.45624550950204,1.51492109047849,1.53945573867507,1.54557845011025,1.54600380297464,1.54557845011025,1.53945573867507,1.51492109047849,1.45624550950204,1.35090946467577;1.52707852123991,1.35090946467577,1.12627027816705,0.873807336098728,0.621344394030401,0.396705207521685,0.220536150957545,0.101463470091274,0.0351361389315130,0.00740198259693121,0.000480822172129528,7.37038817110372e-65,0.000480822172129534,0.00740198259693129,0.0351361389315132,0.101463470091274,0.220536150957545,0.396705207521686,0.621344394030401,0.873807336098728,1.12627027816706,1.35090946467577,1.52707852123991,1.64615120210618,1.71247853326594,1.74021268960053,1.74713385002533,1.74761467219746,1.74713385002533,1.74021268960053,1.71247853326594,1.64615120210618;1.77450847648318,1.64615120210618,1.45624550950204,1.21409026878055,0.941941738241592,0.669793207702639,0.427637966981142,0.237732274377002,0.109375000000000,0.0378758502166066,0.00797914604942120,0.000518313882122886,7.94508807444975e-65,0.000518313882122889,0.00797914604942122,0.0378758502166067,0.109375000000000,0.237732274377002,0.427637966981143,0.669793207702639,0.941941738241592,1.21409026878055,1.45624550950204,1.64615120210618,1.77450847648318,1.84600762626658,1.87590433043376,1.88336516260106,1.88388347648318,1.88336516260106,1.87590433043376,1.84600762626658;1.92038764615428,1.84600762626658,1.71247853326594,1.51492109047849,1.26300884151689,0.979894802046189,0.696780762575492,0.444868513613891,0.247311070826436,0.113781977825802,0.0394019579380968,0.00830064474390017,0.000539197976160105,8.26521449244419e-65,0.000539197976160103,0.00830064474390011,0.0394019579380968,0.113781977825802,0.247311070826436,0.444868513613891,0.696780762575491,0.979894802046190,1.26300884151689,1.51492109047849,1.71247853326594,1.84600762626658,1.92038764615428,1.95148895934848,1.95925040611622,1.95978960409238,1.95925040611622,1.95148895934848;1.98309396859818,1.95148895934848,1.87590433043376,1.74021268960053,1.53945573867507,1.28346368750890,0.995764522514781,0.708065357520657,0.452073306354486,0.251316355429036,0.115624714595798,0.0400400856810820,0.00843507643138297,0.000547930465750837,3.84727135686567e-62,0.000547930465750850,0.00843507643138307,0.0400400856810822,0.115624714595798,0.251316355429037,0.452073306354488,0.708065357520658,0.995764522514781,1.28346368750891,1.53945573867508,1.74021268960053,1.87590433043376,1.95148895934848,1.98309396859818,1.99098111456381,1.99152904502956,1.99098111456381;1.99889962922526,1.99098111456381,1.95925040611622,1.88336516260106,1.74713385002533,1.54557845011025,1.28856826934177,0.999724869457907,0.710881469574045,0.453871288805567,0.252315888890489,0.116084576314754,0.0401993327995967,0.00846862435200551,0.000550109690553512,8.43247710085402e-65,0.000550109690553519,0.00846862435200558,0.0401993327995968,0.116084576314755,0.252315888890490,0.453871288805568,0.710881469574046,0.999724869457908,1.28856826934177,1.54557845011025,1.74713385002533,1.88336516260106,1.95925040611622,1.99098111456381,1.99889962922526,1.99944973891582;];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','AmplitudeSpecification');
            asActual = step(testCase.sbsp,nPoints,4,[]);
            
            % Evaluation
            diff = max(abs(asExpctd(:) - asActual(:))./abs(asExpctd(:)));
            testCase.verifyEqual(asActual,asExpctd,'RelTol',1e-14,...
                sprintf('%g',diff));            
        end
        
        % Test for vertical positive setting
        function testGetPassStopAssignment(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            transition = 0.25;
            direction = Direction.VERTICAL;
            
            % Expected array
            psaExpctd = [
                -1 -1 -1  0  0  0 -1 -1 -1 -1 -1 -1  0 -1 -1 -1 ;
                -1 -1 -1  0  0  0  0 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1  0  0  1  0  0 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1  0  0  1  1  0  0 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1  0  0  1  1  1  0  0 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1  0  0  1  1  1  1  0  0 -1 -1 -1 -1 -1 ;
                -1 -1 -1  0  0  1  1  1  1  1  0  0 -1 -1 -1 -1 ;
                -1 -1 -1  0  0  1  1  1  1  1  1  0  0 -1 -1 -1 ;
                -1 -1 -1 -1  0  0  1  1  1  1  1  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1  0  0  1  1  1  1  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1  0  0  1  1  1  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1  0  0  1  1  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1  0  0  1  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1  0  0  0  0 -1 -1 -1 ;
                -1 -1 -1  0 -1 -1 -1 -1 -1 -1  0  0  0 -1 -1 -1 ;
                -1 -1 -1  0  0 -1 -1 -1 -1 -1 -1  0  0 -1 -1 -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(testCase.sbsp,'Transition',transition);
            psaActual = step(testCase.sbsp,nPoints,1);
            
            % Observation
            %              map = [0 0 0 ; 1 1 1 ; 1 0 0 ; 0 1 0 ; 0 0 1];
            %              subplot(1,2,1); subimage(saExpctd+1,map);
            %              subplot(1,2,2); subimage(saActual+1,map);
            %
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(psaActual, psaExpctd, message);
            
        end
        
        % Test for vertical positive setting
        function testGetPassStopAssignmentHyLx(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            transition = 0.25;
            direction = Direction.VERTICAL;
            
            % Expected array
            psaExpctd = [
                -1 -1 -1 -1  0  0  1  1  1  1  1  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1  0  0  1  1  1  1  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1  0  0  1  1  1  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1  0  0  1  1  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1  0  0  1  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1  0  0  0  0 -1 -1 -1 ;
                -1 -1 -1  0 -1 -1 -1 -1 -1 -1  0  0  0 -1 -1 -1 ;
                -1 -1 -1  0  0 -1 -1 -1 -1 -1 -1  0  0 -1 -1 -1 ;
                -1 -1 -1  0  0  0 -1 -1 -1 -1 -1 -1  0 -1 -1 -1 ;
                -1 -1 -1  0  0  0  0 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1  0  0  1  0  0 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1  0  0  1  1  0  0 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1  0  0  1  1  1  0  0 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1  0  0  1  1  1  1  0  0 -1 -1 -1 -1 -1 ;
                -1 -1 -1  0  0  1  1  1  1  1  0  0 -1 -1 -1 -1 ;
                -1 -1 -1  0  0  1  1  1  1  1  1  0  0 -1 -1 -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'OutputMode','PassStopAssignment');
            set(testCase.sbsp,'Transition',transition);
            psaActual = step(testCase.sbsp,nPoints,2);
            
            % Observation
            %              map = [0 0 0 ; 1 1 1 ; 1 0 0 ; 0 1 0 ; 0 0 1];
            %              subplot(1,2,1); subimage(saExpctd+1,map);
            %              subplot(1,2,2); subimage(saActual+1,map);
            %
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(psaActual, psaExpctd, message);
            
        end
        
        % Test for vertical positive setting
        function testGetPassStopAssignment44(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            dec = [4 4];
            nPoints = [16 16];
            alpha = -1.0;
            transition = 0.25;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            psaExpctd = [
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1  0  0  0  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1  0  1  1  1  0 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1  0  1  1  1  0 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1  0  1  1  1  0 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1  0  0  0  0  0 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec,...
                'OutputMode','PassStopAssignment');
            set(testCase.sbsp,'Transition',transition);
            psaActual = step(testCase.sbsp,nPoints,1);
            
            % Observation
            %              map = [0 0 0 ; 1 1 1 ; 1 0 0 ; 0 1 0 ; 0 0 1];
            %              subplot(1,2,1); subimage(saExpctd+1,map);
            %              subplot(1,2,2); subimage(saActual+1,map);
            %
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(psaActual, psaExpctd, message);
            
        end

        function testGetPassStopAssignmentNext44(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            dec = [4 4];
            nPoints = [16 16];
            alpha = -1.0;
            transition = 0.25;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            psaExpctd = [
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1  0  0  0 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1  0  1  0 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1  0  1  0 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1  0  1  0 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1  0  0  0 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1  0  1  0 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1  0  1  0 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1  0  1  0 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1  0  0  0 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec,...
                'OutputMode','PassStopAssignment');
            set(testCase.sbsp,'Transition',transition);
            psaActual = step(testCase.sbsp,nPoints,2);
            
            % Observation
            %              map = [0 0 0 ; 1 1 1 ; 1 0 0 ; 0 1 0 ; 0 0 1];
            %              subplot(1,2,1); subimage(saExpctd+1,map);
            %              subplot(1,2,2); subimage(saActual+1,map);
            %
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(psaActual, psaExpctd, message);
            
        end

        % To be modified
        function testSubbandAssignment44(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            dec = [4 4];
            nPoints = [16 16];
            alpha = 1.0;
            transition = 0.0;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            saExpctd = [
                11 11  7  7 15 15  3  3  3  3 15 15  7  7 11 11 ;
                11 11 11  7  7 15 15  3  3  3  3 15 15  7  7 11 ;
                6  6 16 16  2  2  4  4 14 14  8  8 10 10 12 12 ;
                12  6  6 16 16  2  2  4  4 14 14  8  8 10 10 12 ;
                12 12  6  6 16 16  2  2  4  4 14 14  8  8 10 10 ;
                10 12 12  6  6 16 16  2  2  4  4 14 14  8  8 10 ;
                5  5 13 13  1  1  1  1 13 13  5  5  9  9  9  9 ;
                9  5  5 13 13  1  1  1  1 13 13  5  5  9  9  9 ;
                9  9  5  5 13 13  1  1  1  1 13 13  5  5  9  9 ;
                9  9  9  5  5 13 13  1  1  1  1 13 13  5  5  9 ;
                8  8 14 14  4  4  2  2 16 16  6  6 12 12 10 10 ;
                10  8  8 14 14  4  4  2  2 16 16  6  6 12 12 10 ;
                10 10  8  8 14 14  4  4  2  2 16 16  6  6 12 12 ;
                12 10 10  8  8 14 14  4  4  2  2 16 16  6  6 12 ;
                7  7 15 15  3  3  3  3 15 15  7  7 11 11 11 11 ;
                11  7  7 15 15  3  3  3  3 15 15  7  7 11 11 11
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec,...
                'OutputMode','SubbandAssignment');
            set(testCase.sbsp,'Transition',transition);
            saActual = step(testCase.sbsp,nPoints);
            
            % Observation
            %              map = [0 0 0 ; 1 1 1 ; 1 0 0 ; 0 1 0 ; 0 0 1];
            %              subplot(1,2,1); subimage(saExpctd+1,map);
            %              subplot(1,2,2); subimage(saActual+1,map);
            %
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(saActual, saExpctd, message);
            
        end

        % Test for vertical positive setting
        function testGetSubbandLocationQuarterSplit44(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            nPoints = [16 16];
            alpha = 1.0;
            direction = Direction.VERTICAL;
            dec = [4 4];
            
            % Expected array
            rosExpctd(:,:,1) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,2) = [
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            
            rosExpctd(:,:,3) = [
                0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,4) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,5) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,6) = [
                0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,7) = [
                0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 0 ];
            
            rosExpctd(:,:,8) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,9) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,10) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,11) = [
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ];
            
            rosExpctd(:,:,12) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 ;
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ;
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,13) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,14) = [
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            
            rosExpctd(:,:,15) = [
                0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 ];
            
            rosExpctd(:,:,16) = [
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 ;
                0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 ;
                0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
            rosExpctd = logical(rosExpctd);
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            spfb = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec,...
                'OutputMode','SubbandLocation',...
                'SplitMode','QuarterSplit');
            
            % Evaluation
            for idx = 1:size(rosExpctd,3)
                rosActual = ...
                    step(spfb,nPoints,idx);
                message = sprintf(...
                    'subband%02d: Actual array is not an expected one.',idx);
                testCase.verifyEqual(rosActual, rosExpctd(:,:,idx),message);
            end
        end

        function testSubbandAssignment44QuarterSplit(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            dec = [4 4];
            nPoints = [16 16];
            alpha = 1.0;
            transition = 0.0;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            saExpctd = [
                11 11  7  7 15 15  3  3  3  3 15 15  7  7 11 11;
                11 11 11  7  7 15 15  3  3  3  3 15 15  7  7 11;
                6  6 14 14  2  2  2  2 14 14  6  6 10 10 10 10;
                10  6  6 14 14  2  2  2  2 14 14  6  6 10 10 10;
                12 12  8  8 16 16  4  4  4  4 16 16  8  8 12 12;
                12 12 12  8  8 16 16  4  4  4  4 16 16  8  8 12;
                5  5 13 13  1  1  1  1 13 13  5  5  9  9  9  9;
                9  5  5 13 13  1  1  1  1 13 13  5  5  9  9  9;
                9  9  5  5 13 13  1  1  1  1 13 13  5  5  9  9;
                9  9  9  5  5 13 13  1  1  1  1 13 13  5  5  9;
                8  8 16 16  4  4  4  4 16 16  8  8 12 12 12 12;
                12  8  8 16 16  4  4  4  4 16 16  8  8 12 12 12;
                10 10  6  6 14 14  2  2  2  2 14 14  6  6 10 10;
                10 10 10  6  6 14 14  2  2  2  2 14 14  6  6 10;
                7  7 15 15  3  3  3  3 15 15  7  7 11 11 11 11;
                11  7  7 15 15  3  3  3  3 15 15  7  7 11 11 11
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec,...
                'OutputMode','SubbandAssignment',...
                'SplitMode','QuarterSplit');
            set(testCase.sbsp,'Transition',transition);
            saActual = step(testCase.sbsp,nPoints);
            
            % Observation
            %              map = [0 0 0 ; 1 1 1 ; 1 0 0 ; 0 1 0 ; 0 0 1];
            %              subplot(1,2,1); subimage(saExpctd+1,map);
            %              subplot(1,2,2); subimage(saActual+1,map);
            %
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(saActual, saExpctd, message);
            
        end
                        
        % Test for vertical positive setting
        function testPassStopAssignment44QuarterSplit(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            dec = [4 4];
            nPoints = [16 16];
            alpha = -1.0;
            transition = 0.25;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            psaExpctd = [
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1  0  0  0  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1  0  1  1  1  0 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1  0  1  1  1  0 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1  0  1  1  1  0 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1  0  0  0  0  0 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec,...
                'OutputMode','PassStopAssignment');
            set(testCase.sbsp,'Transition',transition);
            psaActual = step(testCase.sbsp,nPoints,1);
            
            % Observation
            %              map = [0 0 0 ; 1 1 1 ; 1 0 0 ; 0 1 0 ; 0 0 1];
            %              subplot(1,2,1); subimage(saExpctd+1,map);
            %              subplot(1,2,2); subimage(saActual+1,map);
            %
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(psaActual, psaExpctd, message);
            
        end
        
        function testPassStopAssignment44QuarterSplit3rd(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction        
            dec = [4 4];
            nPoints = [16 16];
            alpha = -1.0;
            transition = 0.25;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            psaExpctd = [
                -1 -1 -1 -1 -1 -1  0  1  1  1  0 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1  0  1  1  1  0 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1  0  0  0  0  0 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1  0  0  0  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1  0  1  1  1  0 -1 -1 -1 -1
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec,...
                'OutputMode','PassStopAssignment',...
                'SplitMode','QuarterSplit');
            set(testCase.sbsp,'Transition',transition);
            psaActual = step(testCase.sbsp,nPoints,3);
            
            % Observation
            %              map = [0 0 0 ; 1 1 1 ; 1 0 0 ; 0 1 0 ; 0 0 1];
            %              subplot(1,2,1); subimage(saExpctd+1,map);
            %              subplot(1,2,2); subimage(saActual+1,map);
            %
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(psaActual, psaExpctd, message);
            
        end
        
        function testPassStopAssignment44QuarterSplit6th(testCase)
            
            % Input argument
            import saivdr.dictionary.utility.Direction
            dec = [4 4];
            nPoints = [16 16];
            alpha = -1.0;
            transition = 0.25;
            direction = Direction.HORIZONTAL;
            
            % Expected array
            psaExpctd = [
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                0 -1 -1 -1  0  0  0 -1 -1 -1 -1 -1 -1 -1  0  0 ;
                -1 -1 -1  0  1  0 -1 -1 -1 -1 -1 -1 -1  0  1  0 ;
                -1 -1  0  0  0 -1 -1 -1 -1 -1 -1 -1  0  0  0 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ;
                -1 -1  0  0  0 -1 -1 -1 -1 -1 -1 -1  0  0  0 -1 ;
                -1  0  1  0 -1 -1 -1 -1 -1 -1 -1  0  1  0 -1 -1 ;
                0  0  0 -1 -1 -1 -1 -1 -1 -1  0  0  0 -1 -1 -1 ;
                -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
                ];
            
            % Instantiation of target class
            import saivdr.dictionary.nsgenlotx.design.*        
            testCase.sbsp = SubbandSpecification(...
                'Alpha',alpha,...
                'Direction',direction,...
                'DecimationFactor',dec,...
                'OutputMode','PassStopAssignment',...
                'SplitMode','QuarterSplit');
            set(testCase.sbsp,'Transition',transition);
            psaActual = step(testCase.sbsp,nPoints,6);
            
            % Observation
            %              map = [0 0 0 ; 1 1 1 ; 1 0 0 ; 0 1 0 ; 0 0 1];
            %              subplot(1,2,1); subimage(saExpctd+1,map);
            %              subplot(1,2,2); subimage(saActual+1,map);
            %
            % Evaluation
            message = 'Actual array is not an expected one.';
            testCase.verifyEqual(psaActual, psaExpctd, message);
            
        end

    end
end
