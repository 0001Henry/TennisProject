import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm
from postprocess import refine_kps
from homography import get_trans_matrix, refer_kps

from tracknet import BallTrackerNet
class CourtDetectorNet():
    def __init__(self, model_type='resnet', device='cuda'):
        self.model_type = model_type
        self.device = device
        
        if self.model_type == 'resnet':
            model = torchvision.models.resnet50()
            model.fc = torch.nn.Linear(model.fc.in_features, 14*2)
            self.model = model
            self.path_model = './models/model_court_det_resnet50.pth'
        elif self.model_type == 'tracknet':
            self.model = BallTrackerNet(out_channels=15)
            self.path_model = './models/model_court_det_tracknet.pt'
        else:
            raise ValueError("model_type should be 'resnet50' or 'tracknet'")
     
        if self.path_model:
            self.model.load_state_dict(torch.load(self.path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
            
    def infer_model(self, frames):
        if self.model_type == 'resnet':
            return self._detect_resnet(frames)
        elif self.model_type == 'tracknet':
            return self._detect_tracknet(frames)
        else:
            raise NotImplementedError(f"No such model: {repr(self.model_type)}")
        
    def _detect_resnet(self, frames):
        output_width = 640
        output_height = 360
        scale = 2
        
        kps_res = []
        matrixes_res = []
        _transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])        
        for _, image in enumerate(tqdm(frames)):
            # img = cv2.resize(image, (224, 224))
            img = _transforms(image)
            img = torch.unsqueeze(img, 0).to(self.device)
            out = self.model(img)
            out = torch.squeeze(out)
            pred = out.detach().cpu().numpy()
            pred[0::2] *= output_width / 112.0
            pred[1::2] *= output_height / 112.0  

            points = []
            # for i in range(0, 28, 2):
            #     x_pred, y_pred = (pred[i],pred[i+1])
            #     if x_pred is not None:
            #         if i not in [8, 12, 9]:
            #             x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred), crop_size=40)
            #         points.append((x_pred, y_pred))                
            #     else:
            #         points.append(None)
            for i in range(0, 28, 2):
                points.append((pred[i],pred[i+1]))

            matrix_trans = get_trans_matrix(points) # matrix_trans.shape=(3, 3) points.shape=(14, 1, 2)
            points = None
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                matrix_trans = cv2.invert(matrix_trans)[1]
            kps_res.append(points)  # points.shape=(14, 1, 2)
            matrixes_res.append(matrix_trans)

        return matrixes_res, kps_res
            
            
    def _detect_tracknet(self, frames):
        output_width = 640
        output_height = 360
        scale = 2
        
        kps_res = []
        matrixes_res = []
        for num_frame, image in enumerate(tqdm(frames)):
            img = cv2.resize(image, (output_width, output_height))
            inp = (img.astype(np.float32) / 255.)
            inp = torch.tensor(np.rollaxis(inp, 2, 0))
            inp = inp.unsqueeze(0) # torch.Size([1, 3, 360, 640])

            out = self.model(inp.float().to(self.device))[0] # torch.Size([15, 360, 640])
            pred = F.sigmoid(out).detach().cpu().numpy() # (15, 360, 640)

            points = []
            for kps_num in range(14):
                heatmap = (pred[kps_num]*255).astype(np.uint8)
                ret, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
                circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2,
                                           minRadius=10, maxRadius=25)
                if circles is not None:
                    x_pred = circles[0][0][0]*scale
                    y_pred = circles[0][0][1]*scale
                    if kps_num not in [8, 12, 9]:
                        x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred), crop_size=40)
                    points.append((x_pred, y_pred))                
                else:
                    points.append(None)

            matrix_trans = get_trans_matrix(points) # matrix_trans.shape=(3, 3) points.shape=(14, 1, 2)
            points = None
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                matrix_trans = cv2.invert(matrix_trans)[1]
            kps_res.append(points)  # points.shape=(14, 1, 2)
            matrixes_res.append(matrix_trans)
            
        return matrixes_res, kps_res     # 变换矩阵和关键点列表        
        
