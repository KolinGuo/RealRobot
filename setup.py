from pathlib import Path
from setuptools import setup, find_packages


def update_repo_root():
    """Updates REPO_ROOT inside _root_dir.py so users do not have to
    specify REAL_ROBOT_ROOT env variable when installed locally"""

    root_dir = Path(__file__).resolve().parent
    with (root_dir / "real_robot/_root_dir.py").open("w") as f:
        f.write(f'REPO_ROOT = "{root_dir}"\n')


if __name__ == "__main__":
    # Update REPO_ROOT inside _root_dir.py
    update_repo_root()

    setup(
        name='real_robot',
        version='0.1.0.dev20230917',
        description="Real Robot xArm7",
        install_requires=[
            'xarm', 'pyrealsense2'
        ],
        packages=find_packages(include=[]),
    )
