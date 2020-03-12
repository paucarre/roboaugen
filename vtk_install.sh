cmake ../ -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=ON \
-DCMAKE_BUILD_TYPE=Release -DVTK_WRAP_PYTHON=ON   -DVTK_PYTHON_VERSION=3  \
-DPYTHON_EXECUTABLE:PATH=$HOME/miniconda2/envs/scara/bin/python \
-DPYTHON_INCLUDE_DIR:PATH=$HOME/miniconda2/envs/scara/include \
-DPYTHON_LIBRARY:PATH=$HOME/miniconda2/envs/roboshape/lib/libpython3.6m.so 