from setuptools import setup

setup(name='svmpy',
      version='0.3',
      description='Naive SVM library in Python',
      long_description=open("README.rst").read(),
      url='https://github.com/dabbabi-eurecom/Python-SVM.git',
      packages=['svmpy'],
      install_requires=[
          'argh',
          'numpy',
          'matplotlib',
          'cvxopt'
      ],
      zip_safe=False)
