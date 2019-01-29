import argparse
import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils import video
import os

CROP_SIZE = 256
DOWNSAMPLE_RATIO = 4

def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def reshape_for_polyline(array):
    """Reshape image so that it works with polyline."""
    return np.array(array, np.int32).reshape((-1, 1, 2))


def resize(image):
    """Crop and resize image for pix2pix."""
    height, width, _ = image.shape
    if height != width:
        # crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size)]
        image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE))
        return image_resize


def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory."""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def main():
    makedirs(args.output_folder)
    # TensorFlow
    graph = load_graph(args.frozen_model_file)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph)

    # OpenCV
    cap = cv2.VideoCapture(args.video_source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    count = 0
    while(cap.isOpened()):
        count = count + 1
        ret, frame = cap.read()

        with open(args.log_file, "a") as file:
            file.write("%d/%d\n" % (count, length))

        # resize image and detect face
        try:
            frame_resize = cv2.resize(frame, None, fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)
        except:
            break
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        black_image = np.zeros(frame.shape, np.uint8)

        for face in faces:
            detected_landmarks = predictor(gray, face).parts()
            landmarks = [[p.x * DOWNSAMPLE_RATIO, p.y * DOWNSAMPLE_RATIO] for p in detected_landmarks]

            jaw = reshape_for_polyline(landmarks[0:17])
            left_eyebrow = reshape_for_polyline(landmarks[22:27])
            right_eyebrow = reshape_for_polyline(landmarks[17:22])
            nose_bridge = reshape_for_polyline(landmarks[27:31])
            lower_nose = reshape_for_polyline(landmarks[30:35])
            left_eye = reshape_for_polyline(landmarks[42:48])
            right_eye = reshape_for_polyline(landmarks[36:42])
            outer_lip = reshape_for_polyline(landmarks[48:60])
            inner_lip = reshape_for_polyline(landmarks[60:68])

            color = (255, 255, 255)
            thickness = 3

            cv2.polylines(black_image, [jaw], False, color, thickness)
            cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
            cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
            cv2.polylines(black_image, [nose_bridge], False, color, thickness)
            cv2.polylines(black_image, [lower_nose], True, color, thickness)
            cv2.polylines(black_image, [left_eye], True, color, thickness)
            cv2.polylines(black_image, [right_eye], True, color, thickness)
            cv2.polylines(black_image, [outer_lip], True, color, thickness)
            cv2.polylines(black_image, [inner_lip], True, color, thickness)

        # generate prediction
        combined_image = np.concatenate([resize(black_image), resize(frame_resize)], axis=1)
        image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
        generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
        image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
        image_normal = np.concatenate([resize(frame_resize), image_bgr], axis=1)
        image_landmark = np.concatenate([resize(black_image), image_bgr], axis=1)

        if args.display_landmark == 0:
            cv2.imwrite(os.path.join(args.output_folder, "scene_%s.png" % '{0:06d}'.format(count)), image_normal)
        else:
            cv2.imwrite(os.path.join(args.output_folder, "scene_%s.png" % '{0:06d}'.format(count)), image_landmark)

    sess.close()
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=str, help='Video file path.')
    parser.add_argument('--show', dest='display_landmark', type=int, default=0, choices=[0, 1],
                        help='0 shows the normal input and 1 the facial landmark.')
    parser.add_argument('--landmark-model', dest='face_landmark_shape_file', type=str, help='Face landmark model file.')
    parser.add_argument('--tf-model', dest='frozen_model_file', type=str, help='Frozen TensorFlow model file.')
    parser.add_argument('-output', '--output-folder', dest='output_folder', type=str, help='Output folder to store image.')
    parser.add_argument('-log', '--log-file', dest='log_file', type=str, help='Log file to know progress.')

    args = parser.parse_args()

    # Create the face predictor and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    main()
