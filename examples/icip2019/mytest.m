import matlab.unittest.TestSuite;

packageSuite  = TestSuite.fromPackage('testcases');

result = run(packageSuite);

disp(result)