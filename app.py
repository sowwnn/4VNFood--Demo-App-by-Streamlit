import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import os
import PIL

import cv2
import torch.nn as nn
import torchvision
from model import *

label = {
    0:"Bánh mì 🌭",
    1:"Cơm tấm 🍛",
    2:"Bánh tráng nướng 🌮",
    3:"Phở 🍜"}
trained_path = "trained"
tasks = [task for task in os.listdir(trained_path) if ("." not in task)]


st.set_page_config(page_title="4VNFood",page_icon="⛰")

def load_image(image_file):
	img = Image.open(image_file).convert('RGB')
	return img


def preprocess(image):
    image = transforms.Resize((224,224))(image)
    image = transforms.ToTensor()(image)
    # image = image/255.0
    return image

def postprocess(image, mask, threshold=0.4):
    mask = mask[0]
    color = {
        0:[0,0,0],
        1:[255,178,102], # Ban mi
        2:[0,55,255], #Pho
        3:[0,2,128], #Comtam
        4:[0,255,128], # Banh trang
    }
    mask = mask.detach().numpy()
    image = image.detach().numpy()
    image = image.transpose(1,2,0)
    w,h,c = image.shape
    label = np.zeros((w,h,c))
    for idx,mas in enumerate(mask):
        for x in range(h):
            for y in range(w):
                if mas[x][y] > threshold:
                    label[x][y] = color[idx]

    label = label/255
    return label
    # label = Image.fromarray(np.uint8(label*255))
    # image = Image.fromarray(np.uint8(image*255))
def match(image,label,alpha=1):
    w,h = image.size
    image = np.array(image)/255
    ovl = np.zeros((w,h,3), dtype=np.uint8)
    ovl = (image) + ( alpha * label)
    ovl = np.where(ovl>=1.0, 0.99, ovl)
    # st.write(ovl.max())
    # st.write(ovl.min())
    return ovl


def predict(image, task, mod):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH = f"{trained_path}/{task}/{mod}"
    # st.write(PATH)
    model = torch.load(PATH, map_location=torch.device('cpu'))
    model.eval()
    # predict = model(image)
    return model(image.unsqueeze(0))


def main():
    st.markdown("<h1 style='text-align: center;'>Phân loại và phân đoạn các món ăn Việt Nam</h1>", unsafe_allow_html=True)
    st.write("")
    st.header("Input 🤌")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file is not None:

            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                            "filesize":image_file.size}
            inpt = load_image(image_file)
            st.image(inpt, width=600)
            w,h = inpt.size
            st.write(inpt.size)

    task = st.radio("Choose your task", tasks)
    models = os.listdir(f"{trained_path}/{task}")
    mod = ""
    if task:
        mod = st.selectbox('Choose ur model', models)

    # st.header("Predict ✍️")
    st.markdown(f"<h1 style='text-align: center;'>Predict 👇</h1>", unsafe_allow_html=True)
    if image_file is not None:
        process = preprocess(inpt)
        # st.write(process.size())
        if task == "segmentation":
            mask = predict(process, task, mod)
            mask = postprocess(process,mask)
            mask = cv2.resize(mask, (w,h), interpolation = cv2.INTER_AREA)
            pred = match(inpt,mask)
            st.image(pred, width=600)
            
        else:
            process = predict(process, task, mod)
            process = process.detach().numpy()
            # st.write(process)
            st.markdown(f"<h1 style='text-align: center;'>{label[np.argmax(process)]}</h1>", unsafe_allow_html=True)
            # st.subheader()


if __name__ == "__main__":
    main()



