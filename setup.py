from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='real_robot',
        version='0.0.2',
        description="Real Robot xArm7",
        install_requires=[
            'xarm', 'pyrealsense2'
        ],
        packages=find_packages(include=[]),
    )
