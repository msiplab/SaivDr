# SaivDr
System object definitions for sparsity-aware image and volume data restoration

## Summary

SaivDr is an abbreviation of Sparsity-Aware Image and Volume Data Restoration. This package is developed in order for making

* Experiments,
* Development and
* Implementation

of sparsity-aware image and volume data restoraition algorithms simple.

Information about SaivDr Package is given in Contents.m. The HELP command can be used to see the contents as follows:

 >> help SaivDr
 
  Sparsity-Aware Image and Volume Data Restoration Package
 
  Files
    mytest     - Script of unit testing for SaivDr Package
    quickstart - Quickstart of *SaivDr Package*
    setpath    - Path setup for *SaivDr Package*
  
* Package structure
  
    + saivdr -+- testcase -+- sparserep 
              |            |
              |            +- embedded
              |            |
              |            +- dictionary  -+- nsolt     -+- design
              |            |               |
              |            |               +- nsoltx    -+- design
              |            |               |
              |            |               +- nsgenlot  -+- design
              |            |               |
              |            |               +- nsgenlotx -+- design
              |            |               |
              |            |               +- olpprfb
              |            |               |
              |            |               +- udhaar 
              |            |               |
              |            |               +- generalfb
              |            |               |
              |            |               +- mixture
              |            |               |
              |            |               +- utility
              |            |
              |            +- restoration -+- ista
              |            |
              |            +- degradation -+- linearprocess
              |            |               |
              |            |               +- noiseprocess
              |            |
              |            +- utility 
              |
              +- sparserep
              |
              +- embedded
              |
              +- dictionary  -+- nsolt     -+- design
              |               |             |
              |               |             +- mexsrcs
              |               |        
              |               +- nsoltx    -+- design
              |               |             |
              |               |             +- mexsrcs
              |               |
              |               +- nsgenlot  -+- design
              |               |         
              |               +- nsgenlotx -+- design
              |               |         
              |               +- olpprfb
              |               |         
              |               +- udhaar 
              |               |
              |               +- generalfb
              |               |
              |               +- mixture
              |               |
              |               +- utility
              |
              +- restoration -+- ista  
              |
              +- degradation -+- linearprocess
              |               |
              |               +- noiseprocess
              |
              +- utility
 
## Requirements
  
* MATLAB R2013b or later, 
** Signal Processing Toolbox
** Image Processing Toolbox
** Optimization Toolbox
  
## Recomendation
  
** Global Optimization Toolbox 
** Parallel Computing Toolbox
** MATLAB Coder
  
## Brief introduction
  
1. Change current directory to where this file contains on MATLAB.
2. Set the path by using the following command:
   >> setpath
3. Several example codes are found under the second layer directory 
   'examples' of this package. Change current directory to one under 
   the second layer directiory 'examples' and execute an M-file of 
   which name begins with 'main,' such as
  
   >> main_xxxx
  
   and then execute an M-file of which name begins with 'disp,' such as
  
   >> disp_xxxx
  
## Contact address
  
     Shogo MURAMATSU,
     Faculty of Engineering, Niigata University,
     8050 2-no-cho Ikarashi, Nishi-ku,
     Niigata, 950-2181, JAPAN
     LinedIn: https://www.linkedin.com/in/shogo-muramatsu-627b084b
  
## References
 
* Shogo Muramatsu, "Structured Dictionary Learning with 2-D Non-
    separable Oversampled Lapped Transform," Proc. of 2014 IEEE 
    International Conference on Acoustics, Speech and Signal Processing
    (ICASSP), pp.2643-2647, May 2014
 
* Kousuke Furuya, Shintaro Hara and Shogo Muramatsu, "Boundary Operation
    of 2-D non-separable Oversampled Lapped Transforms," Proc. of Asia 
    Pacific Signal and Information Proc. Assoc. Annual Summit and Conf.
    (APSIPA ASC), Nov. 2013
 
* Shogo Muramatsu and Natsuki Aizawa, "Image Restoration with 2-D 
    Non-separable Oversampled Lapped Transforms," Proc. of 2013 IEEE 
    International Conference on Image Processing (ICIP), pp.1051-1055, 
    Sep. 2013 
 
* Shogo Muramatsu and Natsuki Aizawa, "Lattice Structures for 2-D 
    Non-separable Oversampled Lapped Transforms, Proc. of 2013 IEEE 
    International Conference on Acoustics, Speech and Signal Processing
    (ICASSP), pp.5632-5636, May 2013 
 
## Acknowledgement
 
    This project was supported by JSPS KAKENHI (23560443,26420347).
 
## Contributors
  
* For coding
** Shintaro HARA,  2013-2014
** Natsuki AIZAWA, 2013-2014
** Kosuke FURUYA,  2013-2015
** Naotaka YUKI,   2014-2015
 
* For testing
** Hidenori WATANABE, 2014
** Kota HORIUCHI,     2015
** Masaki ISHII,      2015
** Takumi KAWAMURA,   2015
** Kenta SEINO,       2015
