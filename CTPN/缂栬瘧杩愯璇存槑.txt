1:caffe通用依赖安装 http://blog.csdn.net/u013832707/article/details/53159071   要使用gpu必须使用cuda及cudnn（cudnn对应版本7.0 v4.0）

  cython安装   sudo apt-get install cython
2:进入code/ctpn/caffe 
  make -j4    4根据自己的内存填写
  <1>:若出现hdf5相关错误 不同系统hdf5位置不一样，修改路径，如我的
    # Whatever else you find you need goes here.
    INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
    LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/ /usr/lib/   x86_64-linux-gnu/hdf5/serial 
 
  make pycaffe 
  
  回到ctpn目录  make   注意目录不能有中文
3: 进入code/ctpn  执行 python  ./tools/demo.py   --no-gpu  (cpu版本)
                       python  ./tools/demo.py             (gpu版本)
   可能出现的问题
   <1>cv2 module找不到 
      解决方法:  sudo apt-get install libopencv-dev python-opencv
   <2>No module named skimage.io
      解决方法:  pip install  scikit-image
   <3>No module named google.protobuf.internal
      解决方法: sudo pip install --upgrade protobuf 
   <4>No module named yaml
      解决方法:sudo apt-get install python-yaml
   <5>libcudnn.so.4 is not a symbolic link
      export PATH=/usr/local/cuda/bin:$PATH'>> ~/.bashrc
      echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
4:输入demo_img文件夹下对应的图片文件名 对该图片处理
