#KLTpicker#

KLT Picker: particle picking using data-driven optimal templates.
Current version: 1.0 
Date: 09/2020

This is the Python version containing the complete source code of the KLT Picker. A MATLAB version is available at http://github.com/amitayeldar/KLTpicker/.

Please cite the following paper when using this package: A. Eldar, B. Landa, and Y. Shkolnisky, "KLT picker: Particle picking using data-driven optimal templates", Journal of Structural Biology, accepted for publication.

##Recommended Environments:
* The package has been tested on Ubuntu 16.04 and Windows 10. It should probably work on other versions of Windows and Linux, but has not been tested on them yet. Similarly for macOS.

* Python 3.6.0+ is required.

* The package makes use of the pyfftw package, which in turn uses the FFTW library. Before installing KLTPicker make sure you have the FFTW library installed on your system: http://www.fftw.org/fftw3_doc/Installation-on-Unix.html#Installation-on-Unix

* For **optional** GPU support, the package requires:
  * NVIDIA CUDA GPU with the Compute Capability 3.0 or larger
  * CUDA Toolkit: v9.0 / v9.2 / v10.0 / v10.1 / v10.2 / v11.0

##Install KLTPicker
###Install KLTPicker via pip:
We recommend to install KLTPicker via pip:

$ pip install kltpicker

In order to enable the GPU support (provided that your system satisfies the above requirements):

$ pip install kltpicker[gpu]

###Install KLTPicker from source
The tarball of the source tree is available via pip download kltpicker. You can install KLTPicker from the tarball:

$ pip install kltpicker-x.x.x.tar.gz

You can also install the development version of KLTPicker from a cloned Git repository:

$ git clone https://github.com/dalitco54/kltpick.git
$ cd kltpicker
$ pip install .

##Uninstall KLTPicker
Use pip to uninstall KLTPicker:

$ pip uninstall kltpicker

##Upgrade KLTPicker
Just use pip with -U option:

$ pip install -U kltpicker

##Getting started:
Please read the user manual for usage instructions
