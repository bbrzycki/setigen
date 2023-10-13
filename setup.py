import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    install_requires = f.readlines()

version_dict = {}
with open("setigen/_version.py") as fp:
    exec(fp.read(), version_dict)
setuptools.setup(
    name="setigen",
    version=version_dict["__version__"],
    author="Bryan Brzycki",
    author_email="bbrzycki@berkeley.edu",
    description="SETI radio signal generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bbrzycki/setigen",
    project_urls={
        "Documentation": "https://setigen.readthedocs.io/en/latest/",
        "Source": "https://github.com/bbrzycki/setigen"
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=["https://github.com/h5py/h5py",
                      "https://github.com/kiyo-masui/bitshuffle"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
