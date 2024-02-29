from setuptools import find_packages, setup

package_name = 'oled_glasses'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hushouyue',
    maintainer_email='hu.shouyue@outllok.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'talker = oled_glasses.publisher_member_function:main',
                'listener = oled_glasses.subscriber_member_function:main',
        ],
    },
)
