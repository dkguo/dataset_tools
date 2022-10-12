conda create --name annotate python==3.6
pip install -r requirements.txt

python3 -c "import OpenGL; print(OpenGL.__path__)"
['/opt/anaconda3/envs/annotate/lib/python3.6/site-packages/OpenGL']

vi /opt/anaconda3/envs/annotate/lib/python3.6/site-packages/OpenGL/platform/ctypesloader.py

#fullName = util.find_library( name )
fullName = '/System/Library/Frameworks/OpenGL.framework/OpenGL'

brew install glew
brew install glfw3

sudo apt-get install libosmesa6-dev
sudo apt-get install freeglut3-dev