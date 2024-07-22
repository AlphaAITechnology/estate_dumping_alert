import numpy as np
import cv2 as cv
import gzip

def get_diff(img_list, human_path_mask = None, threshold=12):
        if not (len(img_list) > 1):
                return None
        
        human_path_mask = np.ones_like(img_list[0]) if human_path_mask is None else human_path_mask
        
        img_bh = cv.GaussianBlur(img_list[0], (15,15), cv.BORDER_DEFAULT)
        img_ah = cv.GaussianBlur(img_list[-1], (15,15), cv.BORDER_DEFAULT)
        img_bh = cv.GaussianBlur(img_bh, (15,15), cv.BORDER_DEFAULT) * human_path_mask
        img_ah = cv.GaussianBlur(img_ah, (15,15), cv.BORDER_DEFAULT) * human_path_mask

        img_bh = cv.cvtColor(img_bh, cv.COLOR_BGR2Lab).astype(np.float32)
        img_ah = cv.cvtColor(img_ah, cv.COLOR_BGR2Lab).astype(np.float32)

        # Bring np.uint8 to proper np.float32 format for CIELab
        img_bh[:,:,1:] -= np.float32(127)
        img_ah[:,:,1:] -= np.float32(127)
        img_bh[:,:,0] *= np.float32(100/255)
        img_ah[:,:,0] *= np.float32(100/255)

        # img_bh = img_bh[:,:,1:]
        # img_ah = img_ah[:,:,1:]

        m_ = np.where(np.sqrt(np.add.reduce(np.square(img_ah - img_bh), axis=2)) >= threshold, 1, 0).astype(np.uint8)
        m_ = cv.GaussianBlur(np.stack((m_, m_, m_), axis=2)*255, (15,15), 0)
        m_ = np.where(m_[:,:,0]>0, 1, 0).astype(np.uint8)

        return np.stack((m_, m_, m_), axis=2)



img_list = [cv.imread("test_base_a.png"),cv.imread("test_base_b.png"), cv.imread("test_base_c.png")]

cv.namedWindow("Live", cv.WINDOW_NORMAL)
cv.imshow("Live", np.vstack((
        get_diff(img_list, None)*255,
        img_list[0],
        img_list[-1]*get_diff(img_list, None),
        img_list[-1],     
)))

cv.waitKey(0)
cv.destroyAllWindows()