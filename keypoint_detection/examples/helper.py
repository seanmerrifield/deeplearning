import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
        last_path = list(train_dir.glob('run_*'))[-1]
        last_run = int(str(last_path).split("_")[-1])
        run_num = last_run + 1

    dir_name = 'run_{:02d}'.format(run_num)
    run_dir = train_dir / dir_name
    run_dir.mkdir()

    return str(run_dir)

