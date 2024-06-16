from setuptools import setup

if __name__ == "__main__":
    setup(
        name="real_robot",
        package_dir={"real_robot": "real_robot"},
        package_data={"real_robot": ["assets/**"]},
        exclude_package_data={"": ["*.convex.stl"]},
        extras_require={
            "xarm": [
                "xArm-Python-SDK @ git+https://github.com/KolinGuo/xArm-Python-SDK.git@custom"
            ]
        },
        zip_safe=False,
    )
