#!/usr/bin/env python
# -*- coding: latin1 -*-

from setuptools import setup, find_packages

ver_dic = {}
version_file = open("loopy/version.py")
try:
    version_file_contents = version_file.read()
finally:
    version_file.close()

exec(compile(version_file_contents, "loopy/version.py", 'exec'), ver_dic)

setup(name="loo.py",
      version=ver_dic["VERSION_TEXT"],
      description="A code generator for array-based code on CPUs and GPUs",
      long_description=open("README.rst", "rt").read(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Other Audience',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.2',
          'Programming Language :: Python :: 3.3',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Software Development :: Libraries',
          'Topic :: Utilities',
          ],

      install_requires=[
          "pytools>=2016.1",
          "pymbolic>=2016.2",
          "genpy>=2016.1.2",
          "cgen>=2016.1",
          "islpy>=2016.2",
          "six>=1.8.0",
          "colorama",
          "Mako",
          ],

      extras_require={
          "pyopencl":  [
              "pyopencl>=2015.2",
              ],
          "fortran":  [
              # Note that this is *not* regular 'f2py2e', this is
              # the Fortran parser from the (unfinished) third-edition
              # f2py, as linked below.
              "f2py>=0.3.1",
              "ply>=3.6",
              ],
          },

      dependency_links=[
          "hg+https://bitbucket.org/inducer/f2py#egg=f2py==0.3.1"
          ],

      scripts=["bin/loopy"],

      author="Andreas Kloeckner",
      url="http://mathema.tician.de/software/loopy",
      author_email="inform@tiker.net",
      license="MIT",
      packages=find_packages(),
      )
