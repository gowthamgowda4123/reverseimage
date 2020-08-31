# reverseimagesearch for video

# currently working on offline mode

Download the project

At videoframe.py 
  cam = cv2.VideoCapture("enter path of video ")
  
  name = 'path to store video frames' + str(currentframe) + '.jpg' # reccomended to give reverseimage static folder path ie.. static/img/*.jpg
  

run videoframe.py to extract frames from the video #("for quick testing use small size video")

install the requirements

run the reverse image search django project
  python manage.py runserver

after the excecution its shows like:
  Starting development server at http://127.0.0.1:8000/
click on the url to perform reverse image search.
  
