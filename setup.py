from setuptools import find_packages, setup

def get_requirements(file_path:str)->list[str]:
    HYPEN_E_DOT="-e ."
    '''
    function will return the requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements



setup(
    name="ML Project",
    version="0.0.1",
    author="Yashodeep More",
    author_email="yashodeepmore01@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)