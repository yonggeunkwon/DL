import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import models
from model import *


class ModelOutputs():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        self.gradients = []
        for name, module in self.model.named_children(): # 모든 layer에 대해서 직접 접근
            x = module(x)
            if name == 'avgpool': # avgpool이후 fully connect하기 전 data shape을 flatten시킴
                x = torch.flatten(x,1)
                
            if name in self.target_layers: # target_layer라면 해당 layer에서의 gradient를 저장
                x.register_hook(self.save_gradient)  
                target_feature_maps = x
            
        return target_feature_maps, x


class GradCam:
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
            
        features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features.cpu().data.numpy()[0, :] # A^k, (2048, 7, 7)

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
    
# def show_cam_on_image(img, mask):

#     # mask = (np.max(mask) - np.min(mask)) / (mask - np.min(mask))
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     cv2.imshow("cam", np.uint8(255 * cam))
#     cv2.imshow("heatmap", np.uint8(heatmap * 255))
#     cv2.waitKey()
    

if __name__ == '__main__':

    def preprocess_image(img):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = img.copy()[:, :, ::-1]
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
        preprocessed_img = \
            np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
        preprocessed_img = torch.from_numpy(preprocessed_img)
        preprocessed_img.unsqueeze_(0)
        input = preprocessed_img.requires_grad_(True)
        return input


    def show_cam_on_image(img, mask):

        # mask = (np.max(mask) - np.min(mask)) / (mask - np.min(mask))
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cv2.imshow("cam", np.uint8(255 * cam))
        cv2.imshow("heatmap", np.uint8(heatmap * 255))
        cv2.waitKey()

    # My Training Model ----------------------------
    model = ResNet50()
    model.load_state_dict(torch.load('./Resnet50_dog_breed_grad.pt', map_location=DEVICE))
    grad_cam = GradCam(model=model, target_layer_names=["layer4"])
    # ----------------------------------------------

    # pretrained resnet50 ------------------------------
    # model_resnet = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2)
    # grad_cam = GradCam(model=model_resnet, target_layer_names=["layer4"])
    # --------------------------------------------------

    # img_path = './dog_images/Afghan_hound/n02088094_1003.jpg'
    # gradcam을 확인하고 싶은 이미지
    img_path = '/home/kwonyonggeun/workspace/DL/stanford_dog_breed/dog_images/Scotch_terrier/n02097298_13186.jpg'
    img = cv2.imread(img_path)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)
    target_index = None
    mask = grad_cam(input, target_index)

    height, width, _ = img.shape
    heatmap = cv2.resize(mask, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    superimposed_img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR)
    superimposed_img = superimposed_img / np.max(superimposed_img)
    superimposed_img = heatmap * 0.3 + superimposed_img * 0.5


    # print(mask)

    # cam = show_cam_on_image(img, mask)
    # print('cam : ', cam)
    cv2.imwrite('/home/kwonyonggeun/workspace/DL/stanford_dog_breed/GRAD-CAM/Scotch_terrier_n02097298_13186.png', np.uint8(255 * superimposed_img))
