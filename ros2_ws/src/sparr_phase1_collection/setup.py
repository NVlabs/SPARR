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


PACKAGE_NAME = "sparr_phase1_collection"


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
        (os.path.join("share", PACKAGE_NAME, "docs"), glob("docs/*.md")),
        (os.path.join("share", PACKAGE_NAME, "schema"), glob("schema/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="SPARR Maintainers",
    maintainer_email="sparr-maintainers@example.com",
    description="SPARR Phase 1 data collection for Flexiv.",
    license="NVIDIA Software and Model Evaluation License",
    entry_points={
        "console_scripts": [
            "sparr_phase1_collector = sparr_phase1_collection.main:collector_main",
            "sparr_phase1_preflight = sparr_phase1_collection.preflight:main",
            "sparr_phase1_validate = sparr_phase1_collection.validation:main",
            "sparr_phase1_recover = sparr_phase1_collection.recover:main",
            "sparr_phase1_summary = sparr_phase1_collection.summary:main",
            "sparr_phase1_replay = sparr_phase1_collection.replay:main",
            "sparr_phase1_camera = sparr_phase1_collection.camera_node:main",
        ],
    },
)
