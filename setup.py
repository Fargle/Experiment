from setuptools import find_packages, setup

setup(
    author="Ian Campbell",
    name="Experiment-and-Analysis",
    version="0.0.1",
    install_requires=[
        "tensorflow",
        "pandas",
        "scikit-learn",
        "numpy",
        "wandb",
        "eli5",
        "h5py",
        "scikeras",
        "pdpbox",
        "hydra-core",
        "streamlit",
    ],
    packages=find_packages(),
)
