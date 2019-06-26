#!/usr/bin/env python3

import setuptools, os, re

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(

    # Basic meta
    name="DA-tutorials",
    version="1.0.0",
    author="Patrick N. Raanes",
    author_email="patrick.n.raanes@gmail.com",
    description="Tutorials on data assimilation (DA)",

    python_requires='~=3.6', # (need >=3.6 for mpl 3.1)
    install_requires=[
      'DA-DAPPER>=0.9.2',
      'jupyter>=1.0.0',
      'Markdown>=2.6',
      ],

    # Note: find_packages() only works on __init__ dirs.
    packages=setuptools.find_packages(),

    # Detailed meta
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',

        'Programming Language :: Python :: 3',

        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nansencenter/DA-tutorials",
    keywords='data-assimilation enkf kalman-filtering state-estimation particle-filter kalman bayesian-methods bayesian-filter chaos',
    # project_urls={
        # 'Documentation': 'https://packaging.python.org/tutorials/distributing-packages/',
        # 'Funding': 'https://donate.pypi.org',
        # 'Say Thanks!': 'http://saythanks.io/to/example',
        # 'Source': 'https://github.com/pypa/sampleproject/',
        # 'Tracker': 'https://github.com/pypa/sampleproject/issues',
    # },
)


