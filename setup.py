from setuptools import setup, find_packages

setup(
    name="bytefield",
    packages=find_packages(exclude=["tests*"]),
    version="1.0.0",
    license="MIT",
    description="Parse binary data using declarative field layout and native Python properties",
    author="donadigo",
    author_email="donadigo@gmail.com",
    url="https://github.com/donadigo/bytefield",
    download_url="https://github.com/donadigo/bytefield",
    keywords=["struct", "bytefield", "bytearray"],
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ]
)
