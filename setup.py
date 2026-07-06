from setuptools import setup, find_packages

setup(
    name="verl",
    version="0.2.0",
    description="Modified verl framework with CORE (Concept-Oriented REinforcement) integration",
    author="Zijun Gao, Zhikun Xu, Xiao Ye, Ben Zhou",
    url="https://github.com/ARC-ASU/CORE",
    packages=find_packages(include=["verl", "verl.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "ray>=2.10.0",
        "hydra-core>=1.3.2",
        "omegaconf",
        "pandas",
        "pyarrow",
        "numpy",
        "tqdm",
    ],
)
