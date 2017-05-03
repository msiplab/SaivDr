 function arrayCoefs = fcn_AtomConcatenationVerticalTypeII( ...
     arrayCoefs, nRows, nCols, paramMtx1,paramMtx2,isPeriodicExt) %#codegen
 %FCN_ATOMCONCATENATIONVERTICALTYPEII
 %
 % SVN identifier:
 % $Id: fcn_AtomConcatenationVerticalTypeII.m 683 2015-05-29 08:22:13Z sho $
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
 persistent h;
 if isempty(h)
     h = saivdr.dictionary.nsolt.mexsrcs.AtomConcatenationVerticalTypeII();
 end
 arrayCoefs = step(h,arrayCoefs,nRows,nCols,paramMtx1,paramMtx2,isPeriodicExt);
 end
