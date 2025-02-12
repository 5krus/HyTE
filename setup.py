"""
Package Setup Settings.

This script manages the details of PyPI deployments. 
"""

# Preparing imports.
import os
from setuptools import setup, find_packages

# Prepare project details.
VERSION = '0.1.3'
DESCRIPTION = 'LLM-based hypothesize-test-evaluate automation.'
LONG_DESCRIPTION = ('A package that allows for LLMs to create hypotheses, test them '
                    'experimentally, and evaluate whether their hypotheses were correct - within '
                    'a loop. As the process loops, the LLMs iterate towards discovey of insights.'
                    )

# Obtain read-me details for long description.
current_location = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_location, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

# Setting up package details.
setup(
    name="HyTE",
    version=VERSION,
    author="5krus (Eryk Krusinski)",
    author_email="<eryk@krus.co.uk>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['openai', 'python-dotenv', 'setuptools', 'wheel'],
    keywords=[
        'llm',
        'optimisation',
        'whittle laboratory',
        'reasoning',
        'experimentation',
        'scientific method'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
