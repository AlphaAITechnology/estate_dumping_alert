from ultralytics import YOLO
import argparse

import cv2 as cv
import numpy as np
import queue
import threading
import time
import datetime
import requests as req
import os
import json
import gzip as gz
import pytz

def Image_Sending(sending_images_q, api_details, shutdown):

    base_url = api_details["server"]["base_url"] # "https://waste-api-mnzypva.alphaaitech.com"
    upload_point = api_details["server"]["upload_point"] # "file"
    email_point = api_details["server"]["email_point"] # "email/send"
    xapitoken = api_details["server"]["x_api_token"] # "zajvak-9zeCvu-taxsyv"


    while (not shutdown):
        while (not sending_images_q.empty()):
            fpath, hpath, date = sending_images_q.get()

            with open(fpath, "rb") as files_:
                # storing file in s3
                response_img = req.post(
                    f"{base_url}/{upload_point}",
                    files = {'file': (fpath, files_, 'image/webp')},
                    headers = {"x-api-token": xapitoken},
                )
            with open(hpath, "rb") as files_:
                # storing file in s3
                response_highlight = req.post(
                    f"{base_url}/{upload_point}",
                    files = {'file': (hpath, files_, 'image/webp')},
                    headers = {"x-api-token": xapitoken},
                )


            os.remove(fpath) # delete stored memory
            os.remove(hpath) # delete stored memory
            del fpath # hotfix to cure memory leak issue
            del hpath # hotfix to cure memory leak issue
            


            if  (response_img.status_code == 201):
                response_1 = json.loads(response_img.text)
                response_2 = json.loads(response_highlight.text)

                img_url = response_1["fileUrl"] if "fileUrl" in response_1 else None
                highlight_url = response_2["fileUrl"] if "fileUrl" in response_2 else None

                if not ((img_url is None) or (highlight_url is None)):
                    response = req.post(
                        f"{base_url}/{email_point}",
                        headers={"x-api-token": "zajvak-9zeCvu-taxsyv"},
                        data={
                            "dataUrl":img_url,
                            "highlightUrl":highlight_url,
                            "reportDateStart": date,
                            "reportDateEnd": date,
                            "totalDetection":1,
                        }
                    )

                    print(
                        "email response:\t",
                        json.loads(
                            response.text
                        )
                    )

            del date # hotfix to cure memory leak issue
        # time.sleep(1) # might have to adjust


def Image_Saving(data_tuple, sending_images_q):
    print("Diag: Saving Called")

    dtm_, img, himg = data_tuple
    fpath = f"./tmp/{dtm_}_i.jpeg"
    hpath = f"./tmp/{dtm_}_h.jpeg"
    cv.imwrite(fpath, img)
    cv.imwrite(hpath, himg)
    sending_images_q.put((fpath, hpath, dtm_))

    del dtm_ # hotfix to cure memory leak issue
    del img  # hotfix to cure memory leak issue
    del himg  # hotfix to cure memory leak issue


def build_human_path_mask(bbox_lists=[], mask=None):

    if (len(bbox_lists)>0) and (mask is not None):
        res = []
        for bboxs in bbox_lists:
            for bbox in bboxs:
                for x1, y1, x2, y2 in bbox.tolist():
                    res.append((int(x1),int(y1)))
                    res.append((int(x1),int(y2)))
                    res.append((int(x2),int(y2)))
                    res.append((int(x2),int(y1)))
        
        hull = cv.convexHull(np.array(res).reshape((-1,2)), returnPoints=True).reshape((-1,2))
        mask = cv.fillPoly(mask, pts=[hull.reshape((-1,2))], color=(255, 255, 255))
    return np.where(mask>0, 1, 0).astype(np.uint8)
        
def analysis_trigger(mask_2d=None, change=10):
    print("Diag: Call For Analysis")
    if mask_2d is None:
        return False
    m = np.where((mask_2d if len(mask_2d.shape)==2 else mask_2d[...,0])>0, 1, 0).astype(np.int64)
    # Match volume of change
    m = m.reshape((-1,))
    return ((np.add.reduce(m)*100)//(m.shape[0])) >= change

def Image_Analysis(collected_images_q, sending_images_q, roi_mask, mask, model, shutdown):

    bg_subtractor = cv.createBackgroundSubtractorMOG2(varThreshold=50)
    
    minimum_confidence = 0.45
    human_seen_flag = False
    human_gone_window = 0
    human_gone_tolerance = 25

    # scene_images_collection = []
    human_images_collection = []

    while (not shutdown):
        while (not collected_images_q.empty()):
            print(f"Diag: Analyse Image")
            dtm_, img = collected_images_q.get()
            
            results = model(img*roi_mask, stream=True, conf=minimum_confidence, classes=[0], device='cuda:1', verbose=False) # looking for people (class 0)
            results = [np.floor(result.boxes.xyxy.cpu().numpy()).astype(np.int16) for result in results] # bring to xyxy numpy


            if sum([r.shape[0] for r in results]) > 0:
                human_seen_flag = True
                human_images_collection.append((img[:,:,:], results))
                
            else:
                if human_seen_flag:
                    human_gone_window += 1
                    if human_gone_window > human_gone_tolerance:
                        human_seen_flag = False
                        human_gone_window = 0
                        
                        fg_mask = bg_subtractor.apply(img[:,:,:]) # mask after differences were found
                        fg_mask = (np.where(fg_mask>0, np.ones_like(fg_mask), np.zeros_like(fg_mask))*255).astype(np.uint8)


                        # Get & Apply human path mask
                        human_path_mask = build_human_path_mask([r for _, r in human_images_collection], np.zeros_like(img)) # build mask using model results

                        fg_mask = fg_mask * mask[:,:,0] # Masking Foreground
                        fg_mask = fg_mask * human_path_mask[:,:,0] # Masking Humans
                        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS, (3,3)), iterations=3)
                        
                        if analysis_trigger(fg_mask): # analyse results
                            m = -1
                            midx = -1 

                            print(f"Diag: Analysis turned true")

                            for idx, (_, hres) in enumerate(human_images_collection): # get best human picture
                                m_ = max([max([abs((y2-y1)*(x2-x1)) for x1, y1, x2, y2 in bbox.tolist()]) for bbox in hres])
                                m, midx = (m_, idx) if m_ > m else (m, midx)
                            
                            print(f"Diag: Best index {midx}; for {m}")
                            Image_Saving((dtm_, img[:,:,:], human_images_collection[midx][0]), sending_images_q)
                            
                        human_images_collection[:] = [] # empty human collection
                else:
                    _ = bg_subtractor.apply(img[:,:,:])
                    
            
            del dtm_
            del img
                
        # time.sleep(1)                





def Image_Reader(video_link, collected_images_q, shutdown):
    if not shutdown:
        cap = cv.VideoCapture(video_link)
        cap_fps = cap.get(cv.CAP_PROP_FPS)

        try:
            grab_failure_tolerance = 15
            grab_failure_counter = 0
            while(cap.isOpened()):
                ret = cap.grab()
                if (ret and collected_images_q.empty()):
                    ret, frame = cap.retrieve()
                    if ret:
                        dtm_ = datetime.datetime.now(pytz.utc).isoformat().split('+')[0]
                        print(f"Diag: {dtm_} Read Image")
                        collected_images_q.put((f"{dtm_}", frame[:,:,:]))
                else:
                    grab_failure_counter += 1
                time.sleep(1.0/cap_fps)
                
                if (grab_failure_counter > grab_failure_tolerance):
                    break
        except Exception as e:
            print(e)
        finally:
            cap.release()
            shutdown = True
        


def get_mask(api_details, camera_choice="Camera_1", ulimit=0, elimit=None):
    mask_file_paths = api_details[camera_choice]["mask_file_paths"]

    hc_mask = None
    for fpath in (mask_file_paths if elimit is None else mask_file_paths[ulimit:elimit]):
        with gz.open(fpath, 'rt') as file:
            if hc_mask is None:
                hc_mask = np.loadtxt(file).astype(np.uint8)
            else:
                hc_mask = np.loadtxt(file).astype(np.uint8) * hc_mask
    hardcoded_mask = np.stack((hc_mask, hc_mask, hc_mask), axis=2)
    return hardcoded_mask


def main():

    model = YOLO("Weights/yolo11n.pt")
    server_api_details={
        "server":{
            "base_url" : "https://waste-api-mnzypva.alphaaitech.com",
            "upload_point" : "file",
            "email_point" : "email/send",
            "x_api_token" : "zajvak-9zeCvu-taxsyv"
        }
    }
    camera_api_details = {
        "Camera_1" : {
            "video_link": "./20240806_1803.mkv",
            # "video_link": "rtsp://admin:hik12345@180.188.143.227:581",
            "mask_file_paths" : [
                "street_container_mask_581.csv.gz",
                "street_backdrop_mask_581.csv.gz"
            ],
        },
        "Camera_2" : {
            "video_link": "rtsp://admin:12345678a@180.188.143.227:580",
            "mask_file_paths" : [
                "street_container_mask_580.csv.gz",
                "street_backdrop_mask_580.csv.gz"
            ],
        },
    }

    #! Declare choice:
    camera_choice = "Camera_1"
    mask = get_mask(camera_api_details, camera_choice)
    region_to_view_mask = get_mask(camera_api_details, camera_choice, ulimit=1, elimit=2) # completely arbritrary; please change as needed
        
    shutdown = False
    collected_images = queue.Queue()
    # saving_images = queue.Queue()
    sending_images = queue.Queue()

    p1 = threading.Thread(target=Image_Reader, args=(camera_api_details[camera_choice]["video_link"], collected_images, shutdown))
    p2 = threading.Thread(target=Image_Analysis, args=(collected_images, sending_images, region_to_view_mask, mask, model, shutdown))
    # p3 = threading.Thread(target=Image_Saving, args=(saving_images, sending_images, shutdown))
    # p4 = threading.Thread(target=Image_Sending, args=(sending_images, server_api_details, shutdown))

    p1.start()
    p2.start()
    # p3.start()
    # p4.start()

    p1.join()
    p2.join()
    # p3.join()
    



if __name__  == "__main__":
    main()
