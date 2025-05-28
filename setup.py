import os
from setuptools import setup, find_packages

def parse_requirements(filename='requirements.txt'):
    """Read requirements.txt, ignore comments and empty lines."""
    here = os.path.abspath(os.path.dirname(__file__))
    reqs_path = os.path.join(here, filename)
    with open(reqs_path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    installs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        installs.append(line)
    return installs

setup(
    name='multi_agentic_rag',
    version='0.1.0',
    author='Taufeeq',
    author_email='taufeeqa413@gmail.com',
    packages=find_packages(exclude=['tests*', 'docker*', '*.egg-info']),
    install_requires=parse_requirements(),

)