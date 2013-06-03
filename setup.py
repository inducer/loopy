#!/usr/bin/env python
# -*- coding: latin1 -*-

import distribute_setup
distribute_setup.use_setuptools()

from setuptools import setup

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    # 2.x
    from distutils.command.build_py import build_py

ver_dic = {}
version_file = open("loopy/version.py")
try:
    version_file_contents = version_file.read()
finally:
    version_file.close()

exec(compile(version_file_contents, "pyopencl/version.py", 'exec'), ver_dic)

setup(name="loo.py",
      version=ver_dic["VERSION_TEXT"],
      description="An automatic loop generator for OpenCL",
      long_description="",
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Other Audience',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Software Development :: Libraries',
          'Topic :: Utilities',
          ],

      install_requires=[
          "pyopencl>=2013.2",
          "pymbolic>=2013.1",
          "cgen",
          "islpy>=2013.2"
          ],

      author="Andreas Kloeckner",
      url="http://pypi.python.org/pypi/pytools",
      author_email="inform@tiker.net",
      license="MIT",
      packages=[
          "loopy",
          "loopy.codegen",
          "loopy.kernel",
          ],

      # 2to3 invocation
      cmdclass={'build_py': build_py})
