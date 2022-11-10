# face-align
Aligns faces to a template picture and create a timelapse video using ffmpeg

Make sure root has:
- `shape_predictor_68_face_landmarks.dat` download and unzip [this model](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)
- `template.jpg` all input imgs will align to the face in this img
- `input/` contains all input imgs as jpgs
- `output/` will contain the algined frames as jpgs

To run:
```
$ python3.10 main.py
```

`output.mp4` will be output into root
