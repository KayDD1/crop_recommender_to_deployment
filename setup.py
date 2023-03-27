from setuptools import find_packages,setup
from typing import List


HYPHEN_DOT ='-e .'
def get_requirements(file_path:str)->List[str]:
    """
        Function returns list of requirements
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if HYPHEN_DOT in requirements:
            requirements.remove(HYPHEN_DOT)
            
    return requirements


setup(
    name="MLPROJECTEND2END",
    version="0.0.1",
    author="Adekunle Adeseye",
    author_email="b1081572@live.tees.ac.uk",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)


