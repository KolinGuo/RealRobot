from setuptools import setup


if __name__ == "__main__":
    setup(
        name='real_robot',
        version='0.1.0rc1',
        description="Real Robot",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        author="Kolin Guo",
        author_email="ruguo@ucsd.edu",
        url="https://github.com/KolinGuo/RealRobot",
        license="MIT",
        keywords="robotics,sensor,visualization,control",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Operating System :: POSIX :: Linux",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Other Audience",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Framework :: Robot Framework :: Library",
            "Framework :: Robot Framework :: Tool",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Education",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Scientific/Engineering",
            "Topic :: Utilities",
        ],
        python_requires=">=3.8",
        install_requires=[
            'pyrealsense2', 'numpy', 'gymnasium', 'transforms3d',
            'urchin',  # loading URDF
            'pynput',  # monitor keyboard event
            'opencv-python', 'opencv-contrib-python',
            'open3d>=0.17.0', 'Pillow', 'scipy',  # for visualization
            'sapien~=3.0.0.dev'  # for sapien.Pose and simsense
        ],
        package_dir={
            "real_robot": "real_robot",
            "xarm": "3rd_party/xArm-Python-SDK/xarm"
        },
        package_data={"real_robot": ["assets/**"]},
        exclude_package_data={"": ["*.convex.stl"]},
    )
