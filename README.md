# SaivDr Package for MATLAB/Simulink [![View SaivDr Package on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://jp.mathworks.com/matlabcentral/fileexchange/45084-saivdr-package)
System object definitions for sparsity-aware image and volumetric data restoration

## Summary

SaivDr is an abbreviation of Sparsity-Aware Image and Volumetric Data Restoration. 
This package is developed for

* Experiments,
* Development and
* Implementation

of sparsity-aware image and volumetric data restoraition algorithms.

In particular, this package provides a rich set of classes related to 
[non-separable oversampled lapped transform ( *NSOLTs* )](https://sigport.org/documents/multidimensional-nonseparable-oversampled-lapped-transforms-theory-and-design) , which allows for convolutional layers with
 
* Parseval tight (paraunitary),
* Symmetric and
* Multiresolution 

properties. For some features, we have prepared custom layer classes with 
Deep Learning Toolbox. It is now easy to incorporate them into flexible 
configurations and parts of your network.

Information about SaivDr Package is given in Contents.m. The HELP command can 
be used to see the contents as follows:

       >> help SaivDr
        
       Sparsity-Aware Image and Volume Data Restoration Package
         
           Files
             mytest     - Script of unit testing for SaivDr Package
             quickstart - Quickstart of *SaivDr Package*
             setpath    - Path setup for *SaivDr Package*
          
           * Package structure
               
               + saivdr -+- testcase -+- dcnn
                         |            |
                         |            +- sparserep 
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
                         |            |               +- olaols
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
                         |            |               |
                         |            |               +- pds
                         |            |               |
                         |            |               +- metricproj
                         |            |               |
                         |            |               +- denoiser
                         |            |
                         |            +- degradation -+- linearprocess
                         |            |               |
                         |            |               +- noiseprocess
                         |            |
                         |            +- utility 
                         |
                         +- dcnn
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
                         |               +- olaols
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
                         |               |
                         |               +- pds
                         |               |
                         |               +- metricproj
                         |               |
                         |               +- denoiser
                         |            
                         +- degradation -+- linearprocess
                         |               |
                         |               +- noiseprocess
                         |
                         +- utility
    
## Requirements
 
* MATLAB R2013b or later. R2021a is recommended.
 * Signal Processing Toolbox
 * Image Processing Toolbox
 * Optimization Toolbox

## Recomendation
 
 * Deep Learning Toolbox
 * Global Optimization Toolbox 
 * Parallel Computing Toolbox
 * MATLAB Coder
 * GPU Coder

## Brief introduction
 
1. Change current directory to where this file contains on MATLAB.
2. Set the path by using the following command:

        >> setpath

3. Build MEX codes if you have MATLAB Coder.

        >> mybuild

4. Several example codes are found under the second layer directory 
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
     http://msiplab.eng.niigata-u.ac.jp/
 
## References

* Genki Fujii, Yuta Yoshida, Shogo Muramatsu, Shunsuke Ono, Samuel Choi, Takeru Ota, 
  Fumiaki Nin, Hiroshi HibinoTitle: "OCT Volumetric Data Restoration with 
  Latent Distribution of Refractive Index," Proc. of 2019 IEEE International 
  Conference on Image Processing (ICIP), pp.764-768, Sept. 2019

* Yuhei Kaneko, Shogo Muramatsu, Hiroyasu Yasuda, Kiyoshi Hayasaka, Yu Otake, 
  Shunsuke Ono, Masahiro Yukawa, "Convolutional-Sparse-Coded Dynamic Mode Decompsition 
  and Its Application to River State Estimation," Proc. of 2019 IEEE International 
  Conference on Acoustics, Speech and Signal Processing (ICASSP), pp.1872-1876, 
  May 2019

* Shogo Muramatsu, Samuel Choi, Shunske Ono, Takeru Ota, Fumiaki Nin, Hiroshi Hibino,
  "OCT Volumetric Data Restoration via Primal-Dual Plug-and-Play Method," Proc. 
  of 2018 IEEE International Conference on Acoustics, Speech and Signal Processing 
  (ICASSP), pp.801-805, Apr. 2018

* Shogo Muramatsu, Kosuke Furuya and Naotaka Yuki, "Multidimensional Nonseparable 
  Oversampled Lapped Transforms: Theory and Design," IEEE Trans. on Signal Process., 
  Vol.65, No.5, pp.1251-1264, DOI:10.1109/TSP.2016.2633240, March 2017. 

* Kota Horiuchi and Shogo Muramatsu, "Fast convolution technique for Non-separable 
  Oversampled Lapped Transforms," Proc. of Asia Pacific Signal and Information Proc. 
  Assoc. Annual Summit and Conf. (APSIPA ASC), Dec. 2016

* Shogo Muramatsu, Masaki Ishii and Zhiyu Chen, "Efficient Parameter Optimization 
  for Example-Based Design of Non-separable Oversampled Lapped Transform," Proc. 
  of 2016 IEEE Intl. Conf. on Image Process. (ICIP),  pp.3618-3622, Sept. 2016

* Shogo Muramatsu, "Structured Dictionary Learning with 2-D Non-separable 
  Oversampled Lapped Transform," Proc. of 2014 IEEE International Conference 
  on Acoustics, Speech and Signal Processing (ICASSP), pp.2643-2647, May 2014
 
* Kousuke Furuya, Shintaro Hara and Shogo Muramatsu, "Boundary Operation of 
  2-D non-separable Oversampled Lapped Transforms," Proc. of Asia Pacific Signal 
  and Information Proc. Assoc. Annual Summit and Conf. (APSIPA ASC), Nov. 2013
 
* Shogo Muramatsu and Natsuki Aizawa, "Image Restoration with 2-D Non-separable 
  Oversampled Lapped Transforms," Proc. of 2013 IEEE International Conference 
  on Image Process. (ICIP), pp.1051-1055, Sep. 2013 
 
* Shogo Muramatsu and Natsuki Aizawa, "Lattice Structures for 2-D Non-separable 
  Oversampled Lapped Transforms," Proc. of 2013 IEEE International Conference 
  on Acoustics, Speech and Signal Process. (ICASSP), pp.5632-5636, May 2013 
 
## Acknowledgement
 
This work was supported by JSPS KAKENHI Grant Numbers JP23560443, JP26420347 and JP19H04135.
 
## Contributors

### Developpers
* Shintaro HARA,  2013-2014
* Natsuki AIZAWA, 2013-2014
* Kosuke FURUYA,  2013-2015
* Naotaka YUKI,   2014-2015
* Yuya KODAMA,    2020-
* Yasas GODAGE,   2021-
 
### Test contributers
* Hidenori WATANABE, 2014-
* Kota HORIUCHI,     2015-
* Masaki ISHII,      2015-
* Takumi KAWAMURA,   2015-
* Kenta SEINO,       2015-
* Satoshi NAGAYAMA,  2017-
* Shota KAYAMORI,    2017-
* Genki FUJII,       2017-
* Naoki YAMAZAKI,    2017-
* Yuhei KANEKO,      2017-
* Nawapan LAOCHAROENSUK, 2019-
* Yusuke ARAI,       2020-
