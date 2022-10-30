import subprocess


def create_video():
    subprocess.run('ffmpeg -framerate 10 -i output/%d.jpg output.mp4')


if __name__ == '__main__':
    create_video()
