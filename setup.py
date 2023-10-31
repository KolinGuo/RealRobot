from setuptools import setup


if __name__ == "__main__":
    setup(
        name='real_robot',
        version='0.1.0b3',
        description="Real Robot xArm7",
        python_requires=">=3.8",
        install_requires=[
            'pyrealsense2', 'numpy', 'gym', 'transforms3d',
            'urchin',  # loading URDF
            'opencv-python', 'open3d', 'Pillow', 'scipy',  # for visualization
            'sapien'  # for sapien.Pose and simsense
        ],
        package_dir={
            "real_robot": "real_robot",
            "xarm": "3rd_party/xArm-Python-SDK/xarm"
        },
        package_data={"real_robot": ["assets/**"]},
        exclude_package_data={"": ["*.convex.stl"]},
    )
