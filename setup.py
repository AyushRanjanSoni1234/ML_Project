from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements mentioned in the requirements.txt file 
    """
    with open(file_path) as f:
        requirements = f.read().splitlines()
    
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
        
    return requirements

setup(
    name = "ml_project",
    version = "0.0.1",
    author = "Ayush Ranjan Soni",
    author_email = "soniayushranjan@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)

