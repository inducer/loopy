#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

ver_dic = {}
version_file = open("loopy/version.py")
try:
    version_file_contents = version_file.read()
finally:
    version_file.close()

os.environ["AKPYTHON_EXEC_IMPORT_UNAVAILABLE"] = "1"
exec(compile(version_file_contents, "loopy/version.py", 'exec'), ver_dic)


# {{{ capture git revision at install time

# authoritative version in pytools/__init__.py
def find_git_revision(tree_root):
    # Keep this routine self-contained so that it can be copy-pasted into
    # setup.py.

    from os.path import join, exists, abspath
    tree_root = abspath(tree_root)

    if not exists(join(tree_root, ".git")):
        return None

    from subprocess import Popen, PIPE, STDOUT
    p = Popen(["git", "rev-parse", "HEAD"], shell=False,
              stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True,
              cwd=tree_root)
    (git_rev, _) = p.communicate()

    import sys
    if sys.version_info >= (3,):
        git_rev = git_rev.decode()

    git_rev = git_rev.rstrip()

    retcode = p.returncode
    assert retcode is not None
    if retcode != 0:
        from warnings import warn
        warn("unable to find git revision")
        return None

    return git_rev


def write_git_revision(package_name):
    from os.path import dirname, join
    dn = dirname(__file__)
    git_rev = find_git_revision(dn)

    with open(join(dn, package_name, "_git_rev.py"), "w") as outf:
        outf.write("GIT_REVISION = %s\n" % repr(git_rev))


write_git_revision("loopy")

# }}}


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
          "pytools>=2018.4",
          "pymbolic>=2019.1",
          "genpy>=2016.1.2",
          "cgen>=2016.1",
          "islpy>=2016.2",
          "six>=1.8.0",
          "codepy>=2017.1",
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
          "git+https://github.com/pearu/f2py.git"
          ],

      scripts=["bin/loopy"],

      author="Andreas Kloeckner",
      url="http://mathema.tician.de/software/loopy",
      author_email="inform@tiker.net",
      license="MIT",
      packages=find_packages(),
      )
