# Scalable Coding


## Setup
```
    # Python
    python -m venv .env
    python -m pip install -r requirements.txt

    # Metrics
    git clone https://git.uni-due.de/ncs/research/pointclouds/metrics.git
    git clone https://git.uni-due.de/ncs/research/pointclouds/pointcloud-data.git data

    # Open3D
    sudo apt-get install libosmesa6-dev
    mkdir dependencies & cd dependencies
    git clone https://github.com/isl-org/Open3D

    cd Open3D
    util/install_deps_ubuntu.sh

    mkdir build && cd build

    cmake -DENABLE_HEADLESS_RENDERING=ON \
                    -DBUILD_GUI=OFF \
                    -DBUILD_WEBRTC=OFF \
                    -DUSE_SYSTEM_GLEW=OFF \
                    -DUSE_SYSTEM_GLFW=OFF \
                    ..

    make -j$(nproc)
    make install-pip-package
```
