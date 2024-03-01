import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="efficientCV",
    version="0.0.3",
    author="Min Khent",
    author_email="minkhent.dev@gmail.com",
    description="Computational efficient models for computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/minkhent/efficientnets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch",
        "numpy",
    ],
    extras_requires={
        "dev": ["twine", "build"],
    },
    python_requires=">=3.10",
)
