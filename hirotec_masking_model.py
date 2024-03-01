import ultralytics
ultralytics.checks()
from ultralytics import YOLO
from IPython.display import display, Image
import os
from PIL import ImageDraw,Image
# from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np
from time import perf_counter

chunked_Masked_image_folder = os.path.join(os.getcwd(),"Dataset_folder/Masked_image")
BS_500_epochs_model = os.path.join(os.getcwd(), "weights/best.pt")
Dataset_folder = os.path.join(os.getcwd(),"Dataset_folder")
Final_masked_image_folder = os.path.join(os.getcwd(),"Dataset_folder/Merged_mask")
model = YOLO(BS_500_epochs_model)


def Making_Mask(img,img_name):
    print("Masking of the Image started")
    
    if img.split(".")[1] == "png" or img.split(".")[1] == "jpg":
        # print(img.shape)
        image_reading = cv2.imread(img)
        # print(image_reading)
        # image_reading = image_reading.resize((1920,1080))
        # results = yolo_trained_with_65_BS(img,imgsz=768)
        results = model.predict(source=image_reading , imgsz=768, show_labels = False, show_conf = False, conf = 0.11, boxes = False)
        # print(results)
        result = results[0]
        masks = result.masks
        if result.masks:
            # print(len(masks))
            mask1 = masks[0]
            mask = masks.cpu().data[0].numpy()
            # polygon = mask1.cpu().xy[0]
            mask_img = Image.fromarray(mask,"I")
            image = mask_img.convert("L")
            # os.chdir("C:/Users/krsou/Downloads/Hirotec/hirotec_results")
            # print(image.size)
            new_image = image.resize((1920,1080))
            new_image=image
            # saving an image with the mask.
            # mask_image_name = img.split(".")[0]+"_mask."+img.split(".")[1]
            mask_image_name = img_name.split(".")[0]+"_mask."+img_name.split(".")[1]
            # print(mask_image_name)
            # image.save("/hirotec_results/{img}".format(img=saving_name))
            
                # print("-----",os.path.join(Inspected_image_folder,mask_image_name))
                # cv2.normalize(mask,dst=None)
                # cv2.imwrite(os.path.join(Inspected_image_folder,mask_image_name),mask_img)
            # print(Masked_image_folder,mask_image_name)
            # new_image.save(f"{Masked_image_folder}/{mask_image_name}")
            new_image.save(f"{chunked_Masked_image_folder}/{mask_image_name}")
        else:
            print("No detecions")
        

def split_image(image_path, chunk_width, chunk_height):
    image = Image.open(image_path)
    img_width, img_height = image.size

    
    num_chunks_x = img_width // chunk_width
    num_chunks_y = img_height // chunk_height

   
    for i in range(num_chunks_x):
        for j in range(num_chunks_y):
            
            bbox = (i * chunk_width, j * chunk_height, (i + 1) * chunk_width, (j + 1) * chunk_height)
            
            chunk = image.crop(bbox)

            chunk.save(f"C:/Users/krsou/Downloads/Hirotec/Hirotec_git/MSL-Bonnet-Inspection/Dataset_folder/chunks_folder/chunk_{i}_{j}.png") # INstead of saving the image masking each chunk and then saving it
            # print("Image saved",os.getcwd())

def merge_images(image_folder, output_path):
    
    # merged_image = Image.new('RGB', (7680, 4320))
    merged_image = Image.new('RGB', (3072, 1792))
    files_list = os.listdir(image_folder)
    # print(files_list)
    chunk_width = 768
    chunk_height = 448

    for i in range(4):
        for j in range(4):
        
            image_path = os.path.join(image_folder, f"chunk_{str(i) +'_'+ str(j)}_mask.png")
            image_name = f"chunk_{str(i) +'_'+ str(j)}_mask.png"
            # print(image_name)
            if str(image_name) in files_list:
                # print("Inside",image_name)
                image_chunk = Image.open(image_path)
            
            # merged_image.paste(image_chunk, (j * chunk_width, i * chunk_height))
                merged_image.paste(image_chunk, (i * chunk_width, j * chunk_height))

    merged_image.save(output_path)


if __name__ == "__main__":
    
    Complete_process_start = perf_counter()
    for i in os.walk(os.path.join(Dataset_folder,"Enhanced_Dataset")):
        for j in i[2]:
            image_path = i[0]+"/"+j
            chunk_width = 1920
            chunk_height = 1080
            One_Image_start = perf_counter()
            split_image(image_path,chunk_width,chunk_height)
            MM_start = perf_counter()
            for k in os.walk(os.path.join(Dataset_folder,"chunks_folder")):
                # print(k[2])
                for l in k[2]:
                    # print(l)
                    chunk_path = k[0]+"/"+l
                    # print(chunk_path)
                    Making_Mask(chunk_path,l)
                    # break
            MM_stop = perf_counter()
            print("Time Taken in Maksing the Image",MM_stop-MM_start)

            masked_image_folder = os.path.join(Dataset_folder,"Masked_image")
            mask_img_name = j.split(".")[0]+"_mask."+j.split(".")[1]
            output_folder = os.path.join(Final_masked_image_folder,mask_img_name)
            print(output_folder)
            merge_images(masked_image_folder, output_folder)
            One_Image_stop = perf_counter()
            print("Per Image Time taken",One_Image_stop-One_Image_start)
    Complete_process_stop = perf_counter()
    print("Complete time taken for 5 Images",Complete_process_stop-Complete_process_start)
              