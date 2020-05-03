from setuptools import setup, find_packages

setup(name='kltpicker',
      version='0.8',
      description='KLT picker',
      url='http://github.com/dalitco54/kltpicker',
      author='Dalit Cohen',
      author_email='dalitcohen@mail.tau.ac.il',
      packages=find_packages(),
      license='MIT',
      install_requires=[
          'numpy',
          'mrcfile',
          'argparse',
          'scipy',
          'pyfftw',
          'tqdm'
      ],
      python_requires='>=3',
      scripts=['bin/KLTPicker.py'],
      zip_safe=False)
