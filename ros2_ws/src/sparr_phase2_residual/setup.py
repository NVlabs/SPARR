# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from glob import glob
import os

from setuptools import find_packages, setup

PACKAGE_NAME = "sparr_phase2_residual"

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test", "test.*")),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [os.path.join("resource", PACKAGE_NAME)],
        ),
        (os.path.join("share", PACKAGE_NAME), ["package.xml"]),
        (os.path.join("share", PACKAGE_NAME, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="SPARR Maintainers",
    maintainer_email="sparr-maintainers@example.com",
    description="SPARR Phase 2 residual training and inference.",
    license="NVIDIA Software and Model Evaluation License",
    entry_points={
        "console_scripts": [
            "sparr_phase2_make_demos = sparr_phase2_residual.demo_export:main",
            "sparr_phase2_collector = sparr_phase2_residual.main:collector_main",
            "sparr_phase2_learner = sparr_phase2_residual.learner_main:main",
            "sparr_phase2_validate = sparr_phase2_residual.validation:main",
        ],
    },
)
