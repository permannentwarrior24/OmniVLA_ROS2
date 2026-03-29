from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'car'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='apollo',
    maintainer_email='apollo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'image_publisher = car.image_publisher_node:main',
            'vllm_ask = car.vllm_ask_node:main',
            "omnivla_node = car.omnivla_vllm_ask_node:main",
            "omnivla_client = car.omnivla_client_node:main",
            "recv_prompt = car.recv_prompt:main",
        ],
    },
)
