import cv2 as cv
import torch
import numpy as np
import queue
import threading
import time
import datetime
import requests as req
import os
import json
import pandas as pd
import gzip



def get_diff(img_list, human_path_mask = None, threshold=None):
        if not (len(img_list) > 1):
                return None
        
        human_path_mask = np.ones_like(img_list[0]) if human_path_mask is None else human_path_mask
        
        img_bh = cv.GaussianBlur(img_list[0], (15,15), cv.BORDER_DEFAULT)
        img_ah = cv.GaussianBlur(img_list[-1], (15,15), cv.BORDER_DEFAULT)
        img_bh = cv.GaussianBlur(img_bh, (15,15), cv.BORDER_DEFAULT) * human_path_mask
        img_ah = cv.GaussianBlur(img_ah, (15,15), cv.BORDER_DEFAULT) * human_path_mask

        img_bh = cv.cvtColor(img_bh, cv.COLOR_BGR2Lab).astype(np.float32)
        img_ah = cv.cvtColor(img_ah, cv.COLOR_BGR2Lab).astype(np.float32)


        # # median of difference in L
        # median_L = np.median((img_ah[:,:,0] - img_bh[:,:,0]).reshape((-1,)))

        # Disconsider the L in CIELab
        img_bh = img_bh[:,:,1:]
        img_ah = img_ah[:,:,1:]

        # Bring np.uint8 to proper np.float32 format for CIELab
        img_bh -= np.float32(127)
        img_ah -= np.float32(127)
                
        two_stack_mask = np.minimum(
                np.minimum(
                        np.abs(img_ah - img_bh),
                        np.abs(img_ah - img_bh + 256)
                ),
                np.abs(img_ah - img_bh - 256)
        )

        threshold = (threshold if threshold is not None else np.square(2.3))
        m_ = np.where(
            (np.where(two_stack_mask[:,:,0] > threshold, 1, 0) + np.where(two_stack_mask[:,:,1] > threshold, 1, 0)) > 0, 1, 0
        ).astype(np.uint8)


        m_ = cv.morphologyEx(m_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT,(3, 3)), iterations=10)
        # m_ = cv.morphologyEx(m_, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(3, 3)), iterations=20)
        return np.stack((m_, m_, m_), axis=2)

def detect(model, img, conf=0.4, classes=[0, 7, 25]):
    # Run inference
    results = model(img)
    res = results.pandas().xyxy[0]

    # 7, 25 --> truck, umbrella
    # 14,15,16 --> bird, cat, dog

    res = res[res["confidence"]>=conf]
    return res[res["class"].isin(classes)] #res[["xmin", "ymin", "xmax", "ymax"]]


def mask_away_obstacles(obstacle_df, mask):
    
    x1,y1,x2,y2 = [int(i) for i in obstacle_df.to_numpy().tolist()[0]]
    
    _, w, _ = mask.shape
    if (((x1+x1)/2)/w) < 0.30: # if obtacle center is in the left thirds of the image
        mask[y1:, x1:x2, ...] = 0 # blackout all the way to the bottom
    else:
        mask[y1:y2, x1:x2, ...] = 0 # normal overlapping blackout

    return mask

def get_human_path_mask(human_path_mask, coor_list=[]):
    human_path_mask = np.uint8(human_path_mask)
    
    coor_list_ = []
    for xyxy in coor_list:
        x1,y1,x2,y2 = [int(i) for i in xyxy]
        coor_list_.append((x1,y1))
        coor_list_.append((x1,y2))
        coor_list_.append((x2,y2))
        coor_list_.append((x2,y1))

    hull_points = cv.convexHull(np.array(coor_list_), returnPoints=True).reshape((-1,2))
    cv.fillPoly(human_path_mask, pts=[hull_points], color=(255,255,255))
    return np.uint8(np.where(human_path_mask==255, 1, 0))


def get_changes_bbox(mask):
    res = []
    x = np.add.reduce(mask, axis=2) == 3
    idx = [i for i in zip(np.argmax(x, axis=1).tolist(), (x.shape[1] - np.argmax(x[:,::-1], axis=1)).tolist())]
    for i, (r1, r2) in enumerate(idx):
        if (r2 <= r1):
            continue
        if (r1==0 and not(x[i, r1])):
            continue
        if (np.bitwise_or.reduce(x[i, r1:r2]) and np.bitwise_not(np.bitwise_and.reduce(x[i, r1:r2]))):
            res.append((r1,r2))

    if len(res)>0:
        x1,x2 = np.min(np.array([i for i, _ in res]).reshape((-1,))), np.max(np.array([i for  _,i in res]).reshape((-1,)))


        x = x.T
        res.clear()
        idx = [i for i in zip(np.argmax(x, axis=1).tolist(), (x.shape[1] - np.argmax(x[:,::-1], axis=1)).tolist())]
        for i, (r1, r2) in enumerate(idx):
            if (r2 <= r1):
                continue
            if (r1==0 and not(x[i, r1])):
                continue
            if (np.bitwise_or.reduce(x[i, r1:r2]) and np.bitwise_not(np.bitwise_and.reduce(x[i, r1:r2]))):
                res.append((r1,r2))
        if len(res)>0:
            y1,y2 = np.min(np.array([i for i, _ in res]).reshape((-1,))), np.max(np.array([i for  _,i in res]).reshape((-1,)))

            return ((x1,y1), (x2,y2))
        else:
            return None
    return None



def Email():
    base_url = "https://waste-api-mnzypva.alphaaitech.com"
    upload_point = "file"
    email_point = "email/send"

    while (elegant_shutdown.empty() or (not elegant_shutdown.get())):
        if (not for_sending.empty()):
            file_path, human_file_path, date_ = for_sending.get()

            with open(file_path, "rb") as files_:
                # storing file
                response_img = req.post(
                    f"{base_url}/{upload_point}",
                    files={'file': (file_path, files_, 'image/webp')},
                    headers={"x-api-token": "zajvak-9zeCvu-taxsyv"},
                )
            with open(human_file_path, "rb") as files_:
                # storing file
                response_highlight = req.post(
                    f"{base_url}/{upload_point}",
                    files={'file': (human_file_path, files_, 'image/webp')},
                    headers={"x-api-token": "zajvak-9zeCvu-taxsyv"},
                )
            # deleting file
            os.remove(file_path)
            os.remove(human_file_path)

            if  (response_img.status_code == 201) and (response_highlight.status_code == 201):
                response_1 = json.loads(response_img.text)
                response_2 = json.loads(response_highlight.text)

                img_url = response_1["fileUrl"] if "fileUrl" in response_1 else None
                highlight_url = response_2["fileUrl"] if "fileUrl" in response_2 else None

                if img_url:
                    response = req.post(
                        f"{base_url}/{email_point}",
                        headers={"x-api-token": "zajvak-9zeCvu-taxsyv"},
                        data={
                            "dataUrl":img_url,
                            "highlightUrl":highlight_url,
                            "reportDateStart": date_,
                            "reportDateEnd": date_,
                            "totalDetection":1,
                        }
                    )

                    print(
                        "email response:\t",
                        json.loads(
                            response.text
                        )
                    )
            del file_path
            del human_file_path
            del date_
        # else:
        #     time.sleep(1)
    elegant_shutdown.put(True)

def SaveToDisk():
    while (elegant_shutdown.empty() or (not elegant_shutdown.get())):
        if (not for_saving.empty()):
            imgs_, timestamp_ = for_saving.get() # stored as (List(Tuple(Tuple(ND.ARRAY, str), Tuple(ND.ARRAY, str))), str)

            for (img, img_path), (himg, himg_path) in imgs_:
                cv.imwrite(img_path, img)

                h, w, _ = himg.shape
                himg = cv.resize(himg, (int(w/3), int(h//3)), cv.INTER_NEAREST)

                cv.imwrite(himg_path, himg)
                for_sending.put((img_path, himg_path, timestamp_))
            del imgs_

        # else:
        #     time.sleep(1)
            
    elegant_shutdown.put(True)




def Analyse():
    # hard coded a mask to remove streets
    with gzip.open("street_container_mask.csv.gz", 'rt') as file:
        hc_mask = pd.read_csv(file, header=None).to_numpy().astype(np.uint8)
        hardcoded_mask = np.stack((hc_mask, hc_mask, hc_mask), axis=2)
    

    max_queue_threshold = 15
    img_list_bh = []
    img_ah_coor = []
    
    seen_flg = False
    frames_since_last_spotted = 0
    frames_since_last_spotted_threshold = 25
    minimum_human_confidence_trigger = 0.4


    last_human_image = None # stores image holding photo of humans


    while (elegant_shutdown.empty() or (not elegant_shutdown.get())):
        if not q.empty():
            img, counter = q.get()
        
            detections = detect(model, img * hardcoded_mask, conf=0.2, classes=[0, 25])
            
            humans = detections[detections["class"] == 0] #looking for human classes=[0]
            obstacles = detections[detections["class"].isin([7,25])] #looking for trucks and umbrellas

            # if humans weren't seen before we want higher confidence to trigger, lower confidence to sustain
            humans = humans[humans["confidence"] >= minimum_human_confidence_trigger]
            # reduce to useful points
            humans = humans[["xmin", "ymin", "xmax", "ymax"]]
            obstacles = obstacles[["xmin", "ymin", "xmax", "ymax"]]

            
            if (humans.empty):
                if (not seen_flg):

                    if len(img_list_bh)>max_queue_threshold:
                        img_list_bh.pop(0)
                    img_list_bh.append(img)


                else:
                    
                    frames_since_last_spotted += 1
                    if (frames_since_last_spotted > frames_since_last_spotted_threshold):
                        seen_flg = False
                        frames_since_last_spotted = 0
                        

                        human_path_mask = np.zeros_like(img)
                        human_path_mask = get_human_path_mask(human_path_mask, img_ah_coor) * hardcoded_mask
                        
                        # black out obstacles
                        if not obstacles.empty:
                            human_path_mask = mask_away_obstacles(obstacles, human_path_mask)

                        img_list_bh.append(img)
                        mask = get_diff(img_list_bh, human_path_mask)


                        if (mask is not None):
                            himg, _ = last_human_image

                            xyxy = get_changes_bbox(mask)
                            if xyxy is not None: # if we don't find a minimum bbox then assume negative results and do nothing
                                (x1,y1), (x2,y2) = xyxy
                                cv.rectangle(img_list_bh[-1], (x1,y1), (x2,y2), (255,0,0), 3)


                                for_saving.put(
                                    (
                                        [
                                            ((img_list_bh[-1], f"tmp/img_{counter:09}.jpeg"), (
                                                
                                                np.vstack((
                                                    img_list_bh[-2], # before anything
                                                    himg, # last human image
                                                    img_list_bh[-1], # immediatly after human left,
                                                    # mask*255,
                                                    # img_list_bh[-1] * mask
                                                ))
                                                
                                                , f"tmp/human_{counter:09}.jpeg")),
                                        ], f"{datetime.datetime.now().isoformat()}"
                                    )
                                )
                                # print(((x1,y1), (x2,y2)))
                    
                        img_list_bh.clear()
                        img_ah_coor.clear()
                        last_human_image = None
                    
            else:
                human_np = humans.to_numpy()
                img_ah_coor.extend(human_np.tolist())

                human_num = human_np.shape[0]
                human_size = np.mean(np.multiply.reduce(human_np[:, 2:] - human_np[:, :2], axis=1).reshape((-1,)))
                
                h_img = img[:,:,:]
                for xyxy in human_np.tolist():
                    x1,y1,x2,y2 = [int(i) for i in xyxy]
                    cv.rectangle(h_img, (x1,y1), (x2,y2), (0,255,0), 5)
                
                if last_human_image is None:
                    last_human_image = (h_img, (human_num, human_size))
                else:
                    _, (hn_, hs_) = last_human_image
                    if hn_ < human_num:
                        last_human_image = (h_img, (human_num, human_size))
                    elif hn_ == human_num:
                        if hs_ <= human_size:
                            last_human_image = (h_img, (human_num, human_size))


                seen_flg = True
                frames_since_last_spotted = 0



            del img
            del counter

            time.sleep(0.2)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    elegant_shutdown.put(True)

def Receive():
    rtsp_url = "rtsp://admin:hik12345@180.188.143.227:581"
    cap = cv.VideoCapture(rtsp_url, cv.CAP_FFMPEG)
    
    ret, frame = cap.read()
    count = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            while not q.empty():
                v_ = q.get()
                del v_

            q.put((frame, count))
            count += 1
            count %= 1000000000
        else:
            cap.release()
            cap = cv.VideoCapture(rtsp_url, cv.CAP_FFMPEG)


        # if cap.get(cv.CAP_PROP_BUFFERSIZE) > 1:
        #     cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    elegant_shutdown.put(True)




q = queue.Queue()
for_saving = queue.Queue()
for_sending = queue.Queue()
elegant_shutdown = queue.Queue()


if __name__=='__main__':
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Can be 'yolov5n' - 'yolov5x6', or 'custom'

    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Analyse)
    p3 = threading.Thread(target=Email)
    p4 = threading.Thread(target=SaveToDisk)
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()