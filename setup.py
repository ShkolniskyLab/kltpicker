from setuptools import setup, find_packages

setup(name='kltpicker',
      version='1.2.1',
      description='KLT picker',
      url='http://github.com/dalitco54/kltpick',
      author='Dalit Cohen',
      author_email='dalitcohen@mail.tau.ac.il',
      packages=find_packages(),
      license='GNU General Public License v3.0',
      entry_points = {
        "console_scripts": ['kltpicker = kltpicker.main:main']
        },
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
      zip_safe=False)
