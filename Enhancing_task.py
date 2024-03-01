import sys
import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact 
from basicsr.archs.rrdbnet_arch import RRDBNet

# Dataset_folder_path = "C:/Users/krsou/Downloads/Hirotec/Hirotec_git/MSL-Bonnet-Inspection/Dataset_folder/Images"
Dataset_folder_path = os.path.join(os.getcwd(),"Dataset_folder/Images")
def load_gan_model(weights):
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    netscale = 4
    upsampler = RealESRGANer(scale=netscale,model_path=weights,dni_weight=None,model=model,tile=0,tile_pad=10,pre_pad=0,half=True,gpu_id=0)   
    print('GAN "Model loaded! and is done!!"')             
    return upsampler



# gan_weight = "C:/Users/krsou/Downloads/Hirotec/weights/realesr-animevideov3.pth"
gan_weight = os.path.join(os.getcwd(), "weights/realesr-animevideov3.pth")
# gan_frame = cv2.imread("C:/Users/krsou/Downloads/Hirotec/hirotec_dataset/img_1.jpg")
# print(os.getcwd())
upsampler  = load_gan_model(weights = gan_weight)

def get_gan_image(upsampler,img,pts):
    
    output, _ = upsampler.enhance(img, outscale=4)
    output = Image.fromarray(output)
    contrast = 0.7
    output = ImageEnhance.Contrast(output).enhance(contrast)
    brightness = 1.2
    output = ImageEnhance.Brightness(output).enhance(brightness)
    output = np.asarray(output)
    # output = cv2.resize(output,(1920,1080))
    return output

def gan_output(gan_frame):
    predicted_frame = get_gan_image(upsampler,img = gan_frame, pts = None)
    return predicted_frame 


# dataset_dir = os.getcwd()
print(os.getcwd())

# Enhanced_image_folder = "C:/Users/krsou/Downloads/Hirotec/Hirotec_git/MSL-Bonnet-Inspection/Dataset_folder/Enhanced_Dataset"
Enhanced_image_folder = os.path.join(os.getcwd(), "Dataset_folder/Enhanced_Dataset")
print(Enhanced_image_folder)
# img_saving_path = os.path.join(Dataset_folder_path,Enhanced_image_folder)
# print(Inspected_image_folder,"xxxx")
if not os.path.exists(Enhanced_image_folder):
    os.mkdir(Enhanced_image_folder)
    print("folder created")
else:
    print("folder already there")

# os.chdir(Dataset_folder_path)
print(os.getcwd())
for img in os.listdir(Dataset_folder_path):
    print(img)
    if img.split(".")[1] == "png" or img.split(".")[1] == "jpg":
        # print(img.shape)
        image_reading = cv2.imread(os.path.join(Dataset_folder_path,img))
        # print(image_reading)
        enhanced_frame = gan_output(image_reading)  
        # cv2.imwrite('enache_image_edi2.jpeg',enhanced_frame)
        enhanced_image_name = img.split(".")[0]+"_enhanced."+img.split(".")[1]
        # Enhanced_image_folder = os.path.join(Dataset_path,"Masked_Images_Enhanced")
        # cv2.imwrite('enache_image_edi2.jpeg',enhanced_frame)
        cv2.imwrite(os.path.join(Enhanced_image_folder,enhanced_image_name),enhanced_frame)