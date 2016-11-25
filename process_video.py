#-*- coding: utf-8 -*-
import os
import sys
import argparse
import time
import cv2
import shutil
import scipy
caffe_root = '../../caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops

BLUE = 0
GREEN = 1
RED = 2
class GaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def blur(img, mask):
    w = img.size[0]
    h = img.size[1]
    mask = cv2.resize(mask, (w, h))
    blue_channel = mask[:, :, BLUE]
    green_channel = mask[:, :, GREEN]

    # Get region to be blurred
    blur_region = np.zeros((h, w))
    blur_region = np.bitwise_and(blue_channel == 0, green_channel < 150)
    blur_region = blur_region[:, :, np.newaxis]
    blur_region = np.tile(blur_region, (1, 1, 3))

    # Blur image
    bimg = img.filter(GaussianBlur(radius=30, bounds=(0, 0, w, h)))

    # Get background and blur foreground
    background = np.multiply(-blur_region, img)
    background = Image.fromarray(background)
    foreground = np.multiply(blur_region, bimg)
    foreground = Image.fromarray(foreground)

    # Merge foreground, background
    img = ImageChops.add(foreground, background, scale=1.0, offset=0)
    return img

def main(argv):
    # Load predicting model
    parser = argparse.ArgumentParser()
    # Optional arguments
    parser.add_argument(
        "--model_def",
        default=os.path.join("/tmp3/changjenyin/trained_models/finetune_bloody_style/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join("/tmp3/changjenyin/trained_models/finetune_bloody_style/finetune_bloody_style_iter_100000.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join("/tmp3/changjenyin/fine_tuning/mean.npy"),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpeg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    # Make classifier
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, gpu=args.gpu, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    # Load fully-convolutional model
    net_full_conv = caffe.Net('/tmp3/changjenyin/caffe/examples/imagenet/bvlc_caffenet_full_conv.prototxt', '/tmp3/changjenyin/caffe/examples/imagenet/bvlc_caffenet_full_conv.caffemodel')
    net_full_conv.set_phase_test()
    net_full_conv.set_mean('data', np.load('/tmp3/changjenyin/fine_tuning/mean.npy'))
    net_full_conv.set_channel_swap('data', (2,1,0))
    net_full_conv.set_raw_scale('data', 255.0)

    read_fifo = open('main_to_script.fifo', 'r')
    write_fifo = open('script_to_main.fifo', 'w')
    while True:
        frame_dir = read_fifo.readline().replace('\n', '')
        if os.path.isdir(frame_dir) == False:
            write_fifo.write("Directory not exists!\n")
            write_fifo.flush()
            continue

        write_fifo.write("got it\n")
        write_fifo.flush()

        print 'Processing ' + frame_dir + '...'
        video_name = os.path.basename(frame_dir)
        output_dir = 'masque/' + video_name
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        classify_time = 0
        heatmap_time = 0
        blur_time = 0
        for img_name in os.listdir(frame_dir):
            # Load net, image
            img_path = frame_dir + '/' + img_name
            img = caffe.io.load_image(img_path)
            img_name_save = os.path.splitext(img_name)[0]

            # Predict image
            inputs = []
            inputs.append(img)

            start = time.time()
            predictions = classifier.predict(inputs)
            end = time.time()
            classify_time += end - start
            if predictions[0].argmax() == 0:
                i = Image.open(img_path)
                i.save(output_dir+'/'+img_name_save+'_output.png')
                continue

            # Make classification map by forward and print prediction indices at each location
            start = time.time()
            out = net_full_conv.forward_all(data=np.asarray([net_full_conv.preprocess('data', img)]))
            end = time.time()
            heatmap_time += end - start

            # Extract classification map as mask
            fig = plt.figure(figsize=(16, 9), frameon=False, dpi=80)
            ax = plt.Axes(fig, [0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)

            mask = output_dir+'/'+img_name_save+'_mask.png'

            plt.imshow(out['prob'][0][1], aspect='normal')
            plt.savefig(mask, dpi=80)

            mask = cv2.imread(mask)

            # Blur image with given mask
            img = Image.open(img_path)
            start = time.time()
            img = blur(img, mask)
            end = time.time()
            blur_time += end - start

            # Output blurred image
            image = output_dir+'/'+img_name_save+'_output.png'
            img.save(image)

        print 'classify_time: ' + str(classify_time) + 'secs'
        print 'heatmap_time: ' + str(heatmap_time) + 'secs'
        print 'blur_time: ' + str(blur_time) + 'secs'
        write_fifo.write("done\n")
        write_fifo.flush()

# Start point if this script is main program
if __name__ == '__main__':
    main(sys.argv)
