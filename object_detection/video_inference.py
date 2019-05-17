import cv2
import os
from pathlib import Path
from inference import Inference

if Path.cwd().name == 'deeplearning':
    os.chdir(str(Path.cwd() / 'object_detection'))


 # Setup paths
MODEL_DIR = str(Path.cwd() / 'models')
DATA_DIR = str(Path.cwd() / 'data')

GRAPH_DIR = str(Path.cwd() / 'models' / 'faster_rcnn_inception_v2_coco_2018_01_28')
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'

VIDEO_DIR = str(Path(DATA_DIR, 'videos'))
LABEL_MAP = str(Path(GRAPH_DIR, 'label_map.pbtxt'))

SAVE_DIR = Path(DATA_DIR, 'videos')
SAVE_DIR.mkdir(parents=True, exist_ok=True)


#Setup Inference instance for object detection
inference = Inference(GRAPH_DIR, MODEL_NAME, LABEL_MAP, n_classes = 90)
# if not inference.has_graph: inference.get_graph(MODEL_NAME) 


for PATH_TO_VIDEO in Path(VIDEO_DIR).glob('*'):
    if not PATH_TO_VIDEO.is_file(): continue
    
    cap = cv2.VideoCapture(str(PATH_TO_VIDEO))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    FILE_OUTPUT = "Detection_" + PATH_TO_VIDEO.name 
    out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      10, (frame_width, frame_height))
 

    while(cap.isOpened()):
        rest, frame = cap.read()

        if rest == True:
            try:
                image, data = inference.detect(image=frame)
                if image is None: continue
            except Exception as e:
                print(e)
                continue

            # Saves for video
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Close window when "Q" button pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()



    # image, data = inference.detect(PATH_TO_IMAGE)
    # if image is None: continue
    # # plt.imshow('Object detector', image)
    # # plt.show()

    # #output image
    # cv.imwrite(str(SAVE_DIR / Path(PATH_TO_IMAGE).name), image)

    # #write out json
    # filename = str(SAVE_DIR / Path(PATH_TO_IMAGE).stem) + '.json'
    # with open(filename, 'w+') as f:
    #     json.dump(data, f)
