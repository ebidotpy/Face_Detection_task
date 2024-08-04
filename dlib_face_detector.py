import argparse
import imutils
import time
import dlib
import cv2
import uuid
from utils.helpers import convert_and_trim_bb

class FaceDetector:
    def __init__(self, model_path, upsample=1):
        print("[INFO] loading CNN face detector...")
        self.detector = dlib.cnn_face_detection_model_v1(model_path)
        self.upsample = upsample

    def load_image(self, image_path):
        print("[INFO] loading image...")
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=600)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, rgb

    def detect_faces(self, rgb_image):
        print("[INFO] performing face detection with dlib...")
        start = time.time()
        results = self.detector(rgb_image, self.upsample)
        end = time.time()
        print("[INFO] face detection took {:.4f} seconds".format(end - start))
        return results

    def draw_boxes(self, image, boxes):
        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image

    def show_image(self, image):
        cv2.imshow("Output", image)
        cv2.waitKey(0)
    
    def save_images(self, image, output_path, boxes):
        for (x, y, w, h) in boxes:
            face = image[y:y + h, x:x + w]
            out_file = f"{output_path}/{uuid.uuid4()}.jpeg"
            cv2.imwrite(out_file, face)

    def run(self, image_path, output_path):
        image, rgb = self.load_image(image_path)
        draw_image = image.copy()
        results = self.detect_faces(rgb)
        boxes = [convert_and_trim_bb(image, r.rect) for r in results]
        draw_image = self.draw_boxes(draw_image, boxes)
        self.save_images(image, output_path, boxes)
        self.show_image(draw_image)

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
                    help="path to input image")
    ap.add_argument("-m", "--model", type=str,
                    default="mmod_human_face_detector.dat",
                    help="path to dlib's CNN face detector model")
    ap.add_argument("-u", "--upsample", type=int, default=1,
                    help="# of times to upsample")
    ap.add_argument("-o", "--output", type=str, required=True, 
                    help="Path to directory for extracted faces")
    args = vars(ap.parse_args())

    face_detector = FaceDetector(model_path=args["model"], upsample=args["upsample"])
    face_detector.run(image_path=args["image"], output_path=args["output"])

if __name__ == "__main__":
    main()