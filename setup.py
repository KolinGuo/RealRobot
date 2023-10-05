from setuptools import setup


if __name__ == "__main__":
    setup(
        name='real_robot',
        version='0.1.0a2',
        description="Real Robot xArm7",
        python_requires=">=3.8",
        install_requires=[
            'xarm', 'pyrealsense2', 'numpy', 'gym', 'transforms3d',
            'urchin',  # loading URDF
            'opencv-python', 'open3d', 'Pillow', 'scipy',  # for visualization
            'sapien'  # for sapien.core.Pose
        ],
        package_data={"real_robot": ["assets/**"]},
        exclude_package_data={"": ["*.convex.stl"]},
    )
