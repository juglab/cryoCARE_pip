import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cryoCARE",
    version="0.0.1",
    author="Tim-Oliver Buchholz",
    author_email="tibuch@mpi-cbg.de",
    description="cryoCARE is deep learning approach for cryo-TEM tomogram denoising.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juglab/cryoCARE_pip",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3",
        "Operating System :: Linux",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "mrcfile",
        "keras>=2.2.4,<2.3.0",
        "tensorflow-gpu>=1.12,<2.0.0",
        "csbdeep>=0.4.0,<0.6.0",
        "PyInquirer"
    ],
    scripts=[
        'cryocare/scripts/cryoCARE_train_data_config.py',
        'cryocare/scripts/cryoCARE_extract_train_data.py',
        'cryocare/scripts/cryoCARE_train_config.py',
        'cryocare/scripts/cryoCARE_train.py',
        'cryocare/scripts/cryoCARE_predict_config.py',
        'cryocare/scripts/cryoCARE_predict.py'
    ]
)
