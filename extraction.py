#Import
import numpy as np
import torch
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import sys
import os

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


#Init args
def parse_args():
    """
    Init args from user and return argparse object 
    """
    parser = argparse.ArgumentParser(description="Extract clothing target / item of interests with SAM")
    parser.add_argument("--data", default="./working", help="Path to target raw images")
    parser.add_argument("--output", default="./output", help="Output path")
    parser.add_argument("--resize", action="store_true", help="Will resize raw input images according to {resize_dim}; helpful if input images is too large by default")
    parser.add_argument("--resize_dim", default='480_720', help="Resize input image to this size; format is W_H")
    parser.add_argument("--pos_keypoint", default='center', help="Location of positive keypoint; modify the script as required if object is not centered across frames")
    parser.add_argument("--enable_neg_keypoints", action="store_true", help="If enabled, will put negative keypoint at specified location via {neg_keypoint}")
    parser.add_argument("--neg_keypoint", default='10_10', help="Location on where to place negative keypoint; make sure coordinate assigned is within the frame; format is W_H")

    args = parser.parse_args()

    return args


#Get args
args = parse_args()


###Initialize SAM
checkpoint = "sam_vit_h_4b8939.pth"
sam_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

#Init base object
sam = sam_model_registry[sam_type](checkpoint=checkpoint)
sam.to(device=device) #put to the specified device

#Init predictor object
predictor = SamPredictor(sam)


#Get image path
imgs = [os.path.join(args.data,i) for i in os.listdir(args.data) if 'jpg' in i or 'png' in i] #get path
imgs.sort() #sort to ensure order is consistent at input and output

#Read 
imgs = [cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2RGB) for i in imgs[:3]] #assumes input is RGB

#Resize if required
if args.resize:
    w,h = args.resize_dim.split("_")
    w,h = int(float(w)),int(float(h))
    
    imgs = [cv2.resize(i,(w,h)) for i in imgs]

    
###Determine keypoints

#Positive keypoint first; assume centered; change it otherwise
h,w,_= imgs[0].shape #ignore channel
k1_h, k1_w = h // 2, w // 2

input_point = np.array([[k1_w, k1_h]])
input_label = np.array([1])

#If negative keypoint is enabled
if args.enable_neg_keypoints:
    k2_w, k2_h = args.neg_keypoint.split("_")
    k2_w,k2_h = int(float(k2_w)),int(float(k2_h))
    
    neg_point = np.array([[k2_w, k2_h]])
    neg_label = np.array([0])
    
    #Append
    input_point = np.concatenate([input_point,neg_point])
    input_label = np.concatenate([input_label,neg_label])
    
    
###Loop

#Holders
all_masks = []
all_isolated = []

for imx,im in enumerate(imgs):

    #Set image
    predictor.set_image(im)

    #Predict
    masks, scores, logits = predictor.predict(
      point_coords=input_point,
      point_labels=input_label,
      multimask_output=True,
    )

    #Select mask; masks seems to go from smallest -> biggest in 3 levels
    #Human segmentation seem to result in clothing being selected in the 2nd level
    #3rd level tend to be entire person, and level 0 seem to be artifact; may be useful for inner clothing

    selected = masks[scores.argmax()] #select maximum score mask; change me as needed
    score_ = scores[scores.argmax()]
    
    #For masks with score that are below a certain threshold, skip it; probably noise
    if score_ < 0.98:
        print("Skipped {}".format(imx)) 
        continue
    
    #Extract RGB masked clothing
    inverse = np.ones(im.shape) - selected[:,:,None] #for making white background
    isolated = (selected[:,:,None] * im) + (inverse*255).astype(np.uint8) #add extra dim to multiply

    #Store
    all_masks.append(selected)
    all_isolated.append(isolated)

    print("Done {}/{}...".format(imx+1,len(imgs)))
    
    
###Save / Write out

#Create dir if not exist
if not os.path.isdir(args.output):
    os.makedirs(args.output)

#Write out
for i in range(len(all_masks)):
    
    #Extract
    b = all_isolated[i]
    b = cv2.cvtColor(b,cv2.COLOR_RGB2BGR) #convert color space

    #Save
    name = "{}/".format(args.output) + "%04d.png" % i
    cv2.imwrite(name,b)

    print("Written {}/{}...".format(i+1,len(all_masks)))
    
print("==="*20)
print("All done!")