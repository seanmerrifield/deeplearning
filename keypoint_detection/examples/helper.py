# import matplotlib
# matplotlib.use('agg')
from highcharts import Highchart
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    for i in range(batch_size):
        plt.figure(figsize=(20, 10))
        ax = plt.subplot(1, batch_size, i + 1)

        # un-transform the image data
        image = test_images[i].data  # get the image from it's Variable wrapper
        image = image.numpy()  # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))  # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints
        predicted_key_pts = predicted_key_pts * 50.0 + 100

        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50.0 + 100

        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)

        plt.axis('off')

    plt.show()



def create_training_dir():

    TRAIN_DIR = './train'
    train_dir = Path(TRAIN_DIR)
    if not train_dir.exists(): train_dir.mkdir(parents=True)

    if len(list(train_dir.glob('run_*'))) == 0:
        run_num = 1
    else:
        last_path = sorted(list(train_dir.glob('run_*')))[-1]
        last_run = int(str(last_path).split("_")[-1])
        run_num = last_run + 1

    dir_name = 'run_{:02d}'.format(run_num)
    run_dir = train_dir / dir_name
    run_dir.mkdir()

    return str(run_dir)

def display_losses(dir):
    assert Path(dir).exists(), "Directory provided doesn't exist"

    chart = create_chart()

    for path in Path(dir).glob('run_*'):
        log_path = Path(path, 'log.txt')
        summary_path = Path(path, 'net_summary.txt')

        losses = log_losses(log_path)
        summary = summary_dict(summary_path)

        batches = np.arange(0, summary['batch_size']*len(losses)*10,  summary['batch_size'])

        data = [[int(x), float(y)] for x,y in zip(batches, losses)]
        chart.add_data_set(data, series_type='line', name="Conv: {},  "
                                        "Out: {},  "
                                        "Kernel: {},  "
                                        "LR: {},  "
                                        "Batch: {},  "
                                        "Drop: {}".format(summary['n_conv'],
                                                          summary['n_full'],
                                                            int(summary['kernel_size']),
                                                            summary['lr'],
                                                            summary['batch_size'],
                                                            summary['p']))
    chart.save_file()
    return chart


def create_chart():

    chart = Highchart()
    options = {
        'title': {
            'text': 'Loss over time for various models'
        },
        'xAxis': {
            'title': {
                'text': 'Number of images'
            },
            'maxPadding': 0.05,
            'showLastLabel': True
        },
        'yAxis': {
            'title': {
                'text': 'Loss'
            },
            'lineWidth': 2
        },
        'legend': {
            'enabled': True,
            'layout': 'vertical',
            'align': 'left',
            'verticalAlign': 'middle'
        },
        'tooltip': {
            'headerFormat': '<b>{series.name}</b><br/>',
            'pointFormat': '{point.x} images: Loss = {point.y}'
        }
    }
    chart.set_dict_options(options)
    return chart

def log_losses(path):
    losses = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line.split(",")) < 2: continue
            for val in line.split(","):
                if len(val.split(": ")) < 2: continue
                k,v = val.split(": ")
                if 'loss' in k.lower(): losses.append(float(v))

    return losses


def summary_dict(path):

    s = {'n_conv': 0, 'n_full': 0}

    with open(path, 'r') as f:

        for line in f.readlines():
            line = line.strip()

            if len(line.split(":")) < 2: continue

            k, v = line.split(":")[0], line.split(":")[1]

            if "learning rate" in k.lower():
                s["lr"] = float(v)
                continue
            if "batch size" in k.lower():
                s["batch_size"] = int(v)
                continue
            if "loss function" in k.lower():
                s["loss"] = v
                continue


            content = re.search('\((.*)\)', v)
            if content == None: continue
            else: content = content.group(1)

            pairs = content.split(",")
            for pair in pairs:
                if len(pair.split("=")) < 2: continue
                p_key, p_val = pair.split("=")
                p_key = p_key.strip()
                if p_key in s: continue

                if p_val[0] == "(": s[p_key] = p_val[1:]
                else: s[p_key] = p_val



            if "conv2d" in line.lower(): s['n_conv'] += 1

            if "full" in line.lower(): s['n_full'] += 1

    if 'p' not in s: s['p'] = 0
    return s

display_losses('./train')