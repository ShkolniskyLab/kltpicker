from setuptools import setup, find_packages

setup(name='kltpicker',
      version='1.2',
      description='KLT picker',
      url='http://github.com/dalitco54/kltpicker',
      author='Dalit Cohen',
      author_email='dalitcohen@mail.tau.ac.il',
      packages=find_packages(),
      license='MIT',
      install_requires=[
          'numpy>=1.16',
          'mrcfile',
          'scipy>=1.3',
          'pyfftw',
          'progressbar2',
          'numba'
      ],
      extras_require={'gpu':['cupy']},
      python_requires='~=3.6',
      scripts=['bin/KLTPicker.py'],
      zip_safe=False)