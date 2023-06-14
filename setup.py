from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
        Desc: This method will read all pypi from requirements remove newline '\n'
    '''
    requirement = []
    with open(file_path) as file_obj:
        requirement = file_obj.readlines()
        requirement = [req.replace('\n','') for req in requirement]
        if HYPHEN_E_DOT in requirement:
            requirement.remove(HYPHEN_E_DOT)
    return requirement

setup(
    name='Breast Cancer Predction',
    version='1.0.0',
    author='Anirudhra',
    author_email='raorudhra16@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt')
    )