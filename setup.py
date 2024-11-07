from setuptools import setup, find_packages

setup(
    name="DiffCSP",  # Name of the package
    version="0.1.0",  # Initial version number
    author="Author Name",
    author_email="author@example.com",
    description="Description of DiffCSP project",
    long_description=open("README.md").read(),  # Optionally read from README.md
    long_description_content_type="text/markdown",
    url="https://github.com/karishmathakrar/DiffCSP_kt",
    packages=find_packages(),  # Automatically finds submodules
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)