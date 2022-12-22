from setuptools import setup

setup(
    name="fetal_brain_utils",
    version="0.0.1",
    packages=["fetal_brain_utils"],
    description="Utilities for fetal brain MRI reconstruction.",
    author="Thomas Sanchez",
    author_email="thomas.sanchez@unil.ch",
    entry_points={
        "console_scripts": [
            "run_mialsrtk = fetal_brain_utils.cli.run_mialsrtk:main",
            "run_niftymic = fetal_brain_utils.cli.run_niftymic:main",
            "run_nesvor = fetal_brain_utils.cli.run_nesvor_from_config:main",
        ],
    },
)
