import os
from subprocess import call
import time
os.mkfifo('main_to_script.fifo')
os.mkfifo('script_to_main.fifo')
write_fifo = open('main_to_script.fifo', 'w')
read_fifo = open('script_to_main.fifo', 'r')

while True:
    video_path = raw_input()
    if not os.path.isfile(video_path):
        print 'The given path is not correct!'
        break

    dir_name = os.path.basename(video_path)
    dir_name = os.path.splitext(dir_name)[0]
    print 'Input video: ' + video_path

    output_dir = 'frames/' + dir_name
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    start = time.time()
    os.system("ffmpeg -i "+video_path+' '+output_dir+'/image-%d.jpeg'+" < /dev/null")
    end = time.time()
    print 'Decoding: ' + str(end - start) + 'secs'

    write_fifo.write(output_dir+"\n")
    '''
    for idx in range(1, len(os.listdir(output_dir))+1):
        frame = os.path.join(output_dir, 'image-' + str(idx) + '.jpeg')

        write_fifo.write(frame+"\n")
    '''
    write_fifo.flush()
    line = read_fifo.readline()
    print line
    line = read_fifo.readline()
    print line

    start = time.time()
    os.system("rm -rf " + dir_name + "_after.mp4")
    os.system("ffmpeg -framerate 12 -i masque/" + dir_name + "/image-%d_output.png " + dir_name +"_after.mp4")
    end = time.time()
    print 'Encoding: ' + str(end - start) + 'secs'
