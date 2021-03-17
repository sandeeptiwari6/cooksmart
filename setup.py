import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cooksmart",
    version="0.0.1",
    author="Sandeep Tiwari",
    author_email="sandy972@uw.edu",
    description="Package for recommending recipes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandeeptiwari6/cooksmart",
    project_urls={
        "Bug Tracker": "https://github.com/sandeeptiwari6/cooksmart/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "cooksmart"},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=['pandas', 'numpy', 'pyLDAvis', 'sklearn', 'plotly',
                      'matplotlib']

)
