from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str) -> List[str]:
    """
    This function reads a requirements file and returns a list of requirements.
    """
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#', '-')]

        return requirements
    
setup(
    name= 'mlproject_ccfd',
    version= '0.0.1',
    author= 'Yogesh Barthwal',
    author_email= 'barthwal.yogesh@gmail.com',
    packages= find_packages(),
    install_requires= get_requirements('requirements.txt'),
    description= 'This is a machine learning project for CCFD',
    long_description= open('README.md'),
    install_requires= 'python>=3.10',
)