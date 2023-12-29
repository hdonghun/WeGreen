# app.py
from flask import Flask, render_template,jsonify
from flask import request

import os, sys
import cv2
from PIL import Image
import numpy as np
import glob
import warnings
import argparse
from cloths_segmentation.pre_trained_models import create_model
import pymysql

# DB 연동 
'''
db_conn = pymysql.connect(
          host = 'localhost',
          port = 3306,
          user = 'root',
          passwd = '1234',
          db = 'test',
          charset= 'utf8'
)

'''


#Flask 객체 인스턴스 생성
app = Flask(__name__)

@app.route('/') # 접속하는 url
def index():
  
  return render_template('firstpage.html')



@app.route('/main') # 접속하는 url
def main():
  
  return render_template('main.html')



@app.route('/chatbot_main') # 접속하는 url
def chatbot():
  
  return render_template('chatbot.html')


@app.route('/login') # 접속하는 url
def login():
  
  return render_template('login.html')


@app.route('/signup') # 접속하는 url
def signup():
  
  return render_template('signup.html')


@app.route('/outer') # 접속하는 url
def outer():
  
  return render_template('outer.html')



@app.route('/cloth', methods=['GET']) # 접속하는 url
def cloth():
  cloth_param = request.args.get('cloth', default=None)
  print(cloth_param)
  return render_template('cloth.html', cloth_param=cloth_param)

@app.route('/fitting') # 접속하는 url
def fitting():
  cloth_param = request.args.get('cloth', default=None)
  print(cloth_param)

  return render_template('fitting.html', cloth_param=cloth_param)

import os
import shutil
@app.route('/upload_selfie', methods=['POST'])
def upload_selfie():
    try:
        # 폼 데이터에서 'selfie' 파일 가져오기
        selfie = request.files['selfie']
        cloth_param = request.args.get('cloth_param', default=None)
        # 저장 디렉토리 지정
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)

        # 파일을 지정된 디렉토리에 저장
        selfie_path = os.path.join(upload_folder, 'user_selfie.jpg')
        selfie.save("/home/alpaco/mento/TryYours-Virtual-Try-On/static/origin_web.jpg")
        
        shutil.copyfile("/home/alpaco/mento/TryYours-Virtual-Try-On/01839_00.jpg", "/home/alpaco/mento/TryYours-Virtual-Try-On/static/cloth_web.jpg")
        
        #==============================
        human_img_path = selfie_path
        cloth_img_path = f"/cloths/{cloth_param}.JPG"
        
        # Read input image
        img=cv2.imread(f"./{human_img_path}") #osh 
        ori_img=cv2.resize(img,(768,1024))
        cv2.imwrite("./origin.jpg",ori_img)

        # Resize input image
        img=cv2.imread('origin.jpg')
        img=cv2.resize(img,(384,512))
        cv2.imwrite('resized_img.jpg',img)

        # Get mask of cloth
        print("Get mask of cloth\n")
        terminnal_command = f"python get_cloth_mask.py --cloth_img_path {cloth_img_path}"  # osh
        os.system(terminnal_command)

        # Get openpose coordinate using posenet
        print("Get openpose coordinate using posenet\n")
        terminnal_command = "python posenet.py" 
        os.system(terminnal_command)

        # Generate semantic segmentation using Graphonomy-Master library
        print("Generate semantic segmentation using Graphonomy-Master library\n")
        os.chdir("./Graphonomy-master")
        terminnal_command ="python exp/inference/inference.py --loadmodel ./inference.pth --img_path ../resized_img.jpg --output_path ../ --output_name /resized_segmentation_img"
        os.system(terminnal_command)
        os.chdir("../")

        # Remove background image using semantic segmentation mask
        mask_img=cv2.imread('./resized_segmentation_img.png',cv2.IMREAD_GRAYSCALE)
        mask_img=cv2.resize(mask_img,(768,1024))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask_img = cv2.erode(mask_img, k)
        img_seg=cv2.bitwise_and(ori_img,ori_img,mask=mask_img)
        back_ground=ori_img-img_seg
        img_seg=np.where(img_seg==0,215,img_seg)
        cv2.imwrite("./seg_img.png",img_seg)
        img=cv2.resize(img_seg,(768,1024))
        cv2.imwrite('./HR-VITON-main/test/test/image/00001_00.jpg',img)

        # Generate grayscale semantic segmentation image
        terminnal_command ="python get_seg_grayscale.py"
        os.system(terminnal_command)

        # Generate Densepose image using detectron2 library
        print("\nGenerate Densepose image using detectron2 library\n")
        terminnal_command ="python detectron2/projects/DensePose/apply_net.py dump detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
        https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
        origin.jpg --output output.pkl -v"
        os.system(terminnal_command)
        terminnal_command ="python get_densepose.py"
        os.system(terminnal_command)

        # Run HR-VITON to generate final image
        print("\nRun HR-VITON to generate final image\n")
        os.chdir("./HR-VITON-main")
        terminnal_command = "python3 test_generator.py --cuda True --test_name test1 --tocg_checkpoint mtviton.pth --gpu_ids 0 --gen_checkpoint gen.pth --datasetting unpaired --data_list t2.txt --dataroot ./test" 
        os.system(terminnal_command)

        # Add Background or Not
        l=glob.glob("./Output/*.png")

        for i in l:
            img=cv2.imread(i)
            cv2.imwrite(i,img)

        os.chdir("../")
        cv2.imwrite("./static/finalimgosh.png", img) # osh

        #==============================


        return render_template('fitting.html')

    except Exception as e:
        # 업로드 중에 오류가 발생한 경우 오류 메시지를 응답
        response_data = {'status': 'error', 'message': str(e)}
        return jsonify(response_data), 500
      
      

@app.route('/chatbot', methods=["GET", "POST"])
def chatbot_Answer():
    # this was written so that if the recieved action was POST.
    if request.method == 'POST':
        # get user question
        user_question = request.form['question']
        
        response = "2조"

        # return generated answer.
    return jsonify({"response": response })

if __name__=="__main__":
  # host 등을 직접 지정하고 싶다면
  app.run(host="0.0.0.0", port="5000", debug=True)

