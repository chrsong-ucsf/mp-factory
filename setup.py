from setuptools import setup, find_packages

setup(
    name="mp-factory",
    version="1.0.0",
    author="Chris Song",
    author_email="chrissong@example.com",
    description="Massive Processing Factory for medical image processing and deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/chrissong/mp-factory",
    packages=find_packages(include=['code*', 'code.*']),
    package_dir={
        'code': 'code',
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "monai>=0.8.0",
        "nibabel>=3.2.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "huggingface_hub>=0.12.0",
    ],
    entry_points={
        'console_scripts': [
            'mp-factory-train=code.training.train_mednext:main',
            'mp-factory-predict=code.prediction.predict_mednext:main',
            'mp-factory-evaluate=code.evaluation.evaluate_gi_masks:main',
        ],
    },
)
