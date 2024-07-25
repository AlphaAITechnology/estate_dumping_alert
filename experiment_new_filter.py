import numpy as np
import cv2 as cv
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

        return np.stack((m_, m_, m_), axis=2)



def mask_edge(img_a, img_b, threshold=70):
    img_a_e, img_b_e = [cv.Canny(cv.cvtColor(img, cv.COLOR_BGR2GRAY), threshold, threshold*2) for img in [
          cv.GaussianBlur(img_, (5,5), cv.BORDER_DEFAULT) for img_ in [img_a, img_b]
          ]]
    
    img_a = cv.resize(img_a_e, None, fx=0.2, fy=0.2, interpolation=cv.INTER_NEAREST)
    img_b = cv.resize(img_b_e, None, fx=0.2, fy=0.2, interpolation=cv.INTER_NEAREST)

    m_ = np.where(img_a_e != img_b_e, 1, 0).astype(np.uint8)
    mask_edge = np.stack((m_, m_, m_), axis=2)

    return mask_edge




with gzip.open("street_container_mask.csv.gz", 'rb') as file:
    m_ = pd.read_csv(file, header=None, delimiter=',').to_numpy().astype(np.uint8)
    mask = np.stack((m_, m_, m_), axis=2)

imgs = [
    cv.imread(img_p) for img_p in [
        "test_base_a.png",
        "test_base_b.png",
        "test_base_c.png"
    ]
]




# cv.namedWindow("Live", cv.WINDOW_NORMAL)
# cv.imshow("Live", mask_edge*255)
# cv.waitKey(0)
# cv.destroyAllWindows()