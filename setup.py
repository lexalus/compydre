# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
            join(dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='compydre',
    version='0.0.1',
    description='An example package. Generated with cookiecutter-pylibrary.',
    long_description='Whatup',
    author='Alexandra Beattie',
    author_email='alexandra.k.beattie@ou.edu',
    url='https://github.com/lexalus/compydre',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Utilities',
    ],
    python_requires='>=3.6, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    install_requires=[
       'numpy', 'pandas', 'rpy2', 'scipy', 'matplotlib', 'networkx'
    ],
    setup_requires=[
        'pytest-runner',
        'pytest',
        'pytest-mock'
    ],
)
