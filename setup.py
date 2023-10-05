from setuptools import setup

setup(
    name="fetal_brain_utils",
    version="0.0.1",
    packages=["fetal_brain_utils"],
    description="Utilities for fetal brain MRI reconstruction.",
    author="Thomas Sanchez",
    author_email="thomas.sanchez@unil.ch",
    install_requires=["pybids", "nibabel"],
    entry_points={
        "console_scripts": [
            "run_mialsrtk = fetal_brain_utils.cli.run_mialsrtk:main",
            "run_niftymic = fetal_brain_utils.cli.run_niftymic:main",
            "run_nesvor_source = fetal_brain_utils.cli.run_nesvor_from_config:main",
            "run_nesvor = fetal_brain_utils.cli.run_nesvor_docker:main",
            "run_nesvor_qc = fetal_brain_utils.cli.run_nesvor_assess:main",
            "run_svrtk = fetal_brain_utils.cli.run_svrtk:main",
            "visualize_sr = fetal_brain_utils.cli.visualize_sr_dirs:main",
            "visualize_lr = fetal_brain_utils.cli.visualize_lr_dirs_slicer:main",
        ],
    },
)
