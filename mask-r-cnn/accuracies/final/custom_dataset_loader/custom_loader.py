import torch
from PIL import Image
import os
import numpy as np 
from pycocotools.coco import COCO
from torch.utils.data import Dataset

class CustomCocoManualDataset(Dataset):
    
    def __init__(self, root, annFile, transforms=None):
        
        self.root = root 
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        
    def __getitem__(self, idx):
        
        #get the image ID and load image metadata 
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]    
        img_path = os.path.join(self.root, img_info['file_name'])
        
        #open the image using PIL
        img = Image.open(img_path).convert("RGB")
        
        #apply transforms to the image (convert to Tensor)
        
        if self.transforms is not None: 
            img = self.transforms(img)
            
        #load annotations for the image (bounding boxes, categories, segmentation)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        #preapre bounding boxes, labels and segmentation marks
        boxes = []
        labels = []
        masks = []
        
        for ann in anns:
            
            #bounding box in COCO Format (x_min, y_min, width, height)
            x_min, y_min, width, height = ann['bbox']
            
            #check if the bounding box is valid (positive width and height)
            if width > 0 and height > 0:
                boxes.append([x_min, y_min, x_min + width, y_min + height])
                labels.append(ann['category_id'])
                
                #segmentation (polygon)
                if 'segmentation' in ann:
                    masks.append(self.coco.annToMask(ann))
                    
        #convert bounding boxes and labels to tensors 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        #efficiently convert the list of masks into a tensor 
        if masks:
            masks = np.stack(masks, axis=0)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        else:
            masks = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
            
        #prepare the target dictionary 
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }
        
        return img, target
    
    def __len__(self):
        return len(self.ids)
    
            
        
        
        
        
        
        
        
            
            
        