import cv2
import torch
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance
from tqdm import tqdm
import torchvision
from court_reference import CourtReference

class PersonDetector():
    def __init__(self, model_type='yolo', device='cuda'):
        self.model_type = model_type.lower()
        self.device = device
        
        if self.model_type == 'fasterrcnn':
            self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.detection_model = self.detection_model.to(device)
            self.detection_model.eval()
        elif self.model_type == 'yolo':
            self.detection_model = YOLO('yolov8x')
            self.detection_model = self.detection_model.to(device)
        else:
            raise ValueError("model_type should be 'fasterrcnn' or 'yolo'")

        self.court_ref = CourtReference()
        self.ref_top_court = self.court_ref.get_court_mask(2)
        self.ref_bottom_court = self.court_ref.get_court_mask(1)
        self.point_person_top = None
        self.point_person_bottom = None
        self.counter_top = 0
        self.counter_bottom = 0
        
        
    def detect(self, image, person_min_score=0.7):
        if self.model_type == 'fasterrcnn':
            return self._detect_fasterrcnn(image, person_min_score)
        elif self.model_type == 'yolo':
            return self._detect_yolo(image, person_min_score)
        else:
            raise NotImplementedError(f"No such model: {repr(self.model_type)}")

    def _detect_fasterrcnn(self, image, person_min_score=0.7):
        PERSON_LABEL = 1
        frame_tensor = image.transpose((2, 0, 1)) / 255
        frame_tensor = torch.from_numpy(frame_tensor).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            preds = self.detection_model(frame_tensor)

        persons_boxes = []
        probs = []
        for box, label, score in zip(preds[0]['boxes'][:], preds[0]['labels'], preds[0]['scores']):
            if label == PERSON_LABEL and score > person_min_score:
                persons_boxes.append(box.detach().cpu().numpy())
                probs.append(score.detach().cpu().numpy())
        return persons_boxes, probs

    def _detect_yolo(self, image, person_min_score=0.6):
        PERSON_LABEL = 0  # YOLOv8 模型中 'person' 的标签为 0
        results = self.detection_model(image)
        preds = results[0]

        persons_boxes = []
        probs = []
        for box in preds.boxes:
            if box.cls == PERSON_LABEL and box.conf > person_min_score:
                persons_boxes.append(box.xyxy.cpu().numpy()[0])
                probs.append(box.conf.cpu().numpy()[0])
        return persons_boxes, probs

    def detect_top_and_bottom_players(self, image, inv_matrix, filter_players=False):
        matrix = cv2.invert(inv_matrix)[1]
        mask_top_court = cv2.warpPerspective(self.ref_top_court, matrix, image.shape[1::-1])
        mask_bottom_court = cv2.warpPerspective(self.ref_bottom_court, matrix, image.shape[1::-1])
        person_bboxes_top, person_bboxes_bottom = [], []

        bboxes, probs = self.detect(image, person_min_score=0.7 if self.model_type == 'fasterrcnn' else 0.6)
        if len(bboxes) > 0:
            person_points = [[int((bbox[2] + bbox[0]) / 2), int(bbox[3])] for bbox in bboxes]
            person_bboxes = list(zip(bboxes, person_points))

            person_bboxes_top = [pt for pt in person_bboxes if mask_top_court[pt[1][1] - 1, pt[1][0]] == 1]
            person_bboxes_bottom = [pt for pt in person_bboxes if mask_bottom_court[pt[1][1] - 1, pt[1][0]] == 1]

            if filter_players:
                person_bboxes_top, person_bboxes_bottom = self.filter_players(person_bboxes_top, person_bboxes_bottom, matrix)
        return person_bboxes_top, person_bboxes_bottom

    def filter_players(self, person_bboxes_top, person_bboxes_bottom, matrix):
        refer_kps = np.array(self.court_ref.key_points[12:], dtype=np.float32).reshape((-1, 1, 2))
        trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
        center_top_court = trans_kps[0][0]
        center_bottom_court = trans_kps[1][0]
        if len(person_bboxes_top) > 1:
            dists = [distance.euclidean(x[1], center_top_court) for x in person_bboxes_top]
            ind = dists.index(min(dists))
            person_bboxes_top = [person_bboxes_top[ind]]
        if len(person_bboxes_bottom) > 1:
            dists = [distance.euclidean(x[1], center_bottom_court) for x in person_bboxes_bottom]
            ind = dists.index(min(dists))
            person_bboxes_bottom = [person_bboxes_bottom[ind]]
        return person_bboxes_top, person_bboxes_bottom

    def track_players(self, frames, matrix_all, filter_players=False):
        persons_top = []
        persons_bottom = []
        min_len = min(len(frames), len(matrix_all))
        for num_frame in tqdm(range(min_len)):
            img = frames[num_frame]
            if matrix_all[num_frame] is not None:
                inv_matrix = matrix_all[num_frame]
                person_top, person_bottom = self.detect_top_and_bottom_players(img, inv_matrix, filter_players)
            else:
                person_top, person_bottom = [], []
            persons_top.append(person_top)
            persons_bottom.append(person_bottom)
        return persons_top, persons_bottom
