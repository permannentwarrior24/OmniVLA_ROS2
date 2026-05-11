from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mpc_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ('share/ament_index/resource_index/packages',
        #     ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/**')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('goal/*.rviz')),
        (os.path.join('share', package_name, 'goal'), glob('goal/*.txt')),
    ],
    install_requires=['setuptools', 'numpy', 'cvxpy'],
    zip_safe=True,
    maintainer='phak',
    maintainer_email='2967306689@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mpc_controller = mpc_planner.mpc_controller:main',
            'goal_sender = mpc_planner.goal_sender:main',
            'simple_simulator = mpc_planner.simple_simulator:main',
        ]
    },
)
