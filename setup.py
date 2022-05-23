import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cryoCARE_mpido",
    version="0.2.1",
    author="Thorsten Wagner",
    author_email="thorsten.wagner@mpi-dortmund.mpg.de",
    description="cryoCARE is a deep learning approach for cryo-TEM tomogram denoising.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thorstenwagner/cryoCARE_pip",
    packages=setuptools.find_packages(),
    extras_require={
        'c11': ['nvidia-tensorflow[horovod] < 1.16, >= 1.15.5+nv22.1'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License"
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "mrcfile",
        "keras>=2.1.2,<2.4.0",
        "csbdeep>=0.6.0,<0.7.0",
        "psutil"
    ],
    scripts=[
        'cryocare/scripts/cryoCARE_extract_train_data.py',
        'cryocare/scripts/cryoCARE_train.py',
        'cryocare/scripts/cryoCARE_predict.py'
    ]
)
