#### Download following files
##### https://drive.google.com/file/d/137uO2Ya63cpf1RspiE3rbsAE-btkIovG/view?usp=sharing
##### https://drive.google.com/file/d/1PTNmhd-eSq0fwSPv0nvQN8h_scR1v-UJ/view?usp=sharing

#### Unzip files and move models to data folder. The directory should look like this
```
- data
    - scene_221012114441
    - models
```

#### Clone this git
```
git clone https://github.com/dkguo/dataset_tools
```

#### Change dataset_path in config.py to the path of data folder
```
dataset_path = 'YOUR_PATH/data'
```

#### Create environment and install necessary packages
```
conda create --name annotate python==3.6
pip install -r requirements.txt
```

##### If you are are on MacOS, you may need code below to install pyOpenGL properly
```
brew install glew
brew install glfw3

# get path of OpenGL
python3 -c "import OpenGL; print(OpenGL.__path__)"

# you will get a path like this
['/opt/anaconda3/envs/annotate/lib/python3.6/site-packages/OpenGL']

# open YOUR_OPENGL_PATH_ABOVE/platform/ctypesloader.py
vi /opt/anaconda3/envs/annotate/lib/python3.6/site-packages/OpenGL/platform/ctypesloader.py

# change code line 35
# old code
#fullName = util.find_library( name )
# new code
fullName = '/System/Library/Frameworks/OpenGL.framework/OpenGL'
```

##### If you are on Ubuntu, you may following packages to run pyOpenGL
```
sudo apt-get install libosmesa6-dev freeglut3-dev
```
