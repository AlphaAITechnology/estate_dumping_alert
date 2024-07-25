import numpy as np
import cv2 as cv
import pandas as pd
import gzip 

def get_diff(img_list, human_path_mask = None, threshold=10):
        if not (len(img_list) > 1):
                return None
        
        human_path_mask = np.ones_like(img_list[0]) if human_path_mask is None else human_path_mask
        
        img_bh = cv.GaussianBlur(img_list[0], (15,15), cv.BORDER_DEFAULT)
        img_ah = cv.GaussianBlur(img_list[-1], (15,15), cv.BORDER_DEFAULT)
        img_bh = cv.GaussianBlur(img_bh, (15,15), cv.BORDER_DEFAULT) * human_path_mask
        img_ah = cv.GaussianBlur(img_ah, (15,15), cv.BORDER_DEFAULT) * human_path_mask

        img_bh = cv.cvtColor(img_bh, cv.COLOR_BGR2Lab).astype(np.float32)[:,:,1:] - 127
        img_ah = cv.cvtColor(img_ah, cv.COLOR_BGR2Lab).astype(np.float32)[:,:,1:] - 127

        m_ = np.where(np.sqrt(np.add.reduce(np.square(img_ah - img_bh), axis=2)) >= threshold, 1, 0).astype(np.uint8)
        m_ = cv.GaussianBlur(np.stack((m_, m_, m_), axis=2)*255, (15,15), 0)
        m_ = np.where(m_[:,:,0]>0, 1, 0).astype(np.uint8)

        return np.stack((m_, m_, m_), axis=2)






def mask_diff(img_a, img_b, threshold=70):
    img_a_, img_b_ = [cv.GaussianBlur(img_, (5,5), cv.BORDER_DEFAULT) for img_ in [img_a, img_b]]

    img_a_e = cv.Canny(cv.cvtColor(img_a_, cv.COLOR_BGR2GRAY), threshold, threshold*2)
    img_b_e = cv.Canny(cv.cvtColor(img_b_, cv.COLOR_BGR2GRAY), threshold, threshold*2)

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

diff = get_diff(imgs)
mask_edge = mask_diff(imgs[0]*diff, imgs[2]*diff) * 255
mask_edge_ = cv.morphologyEx(mask_edge, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (5,5)), iterations=1)
mask_edge_ = cv.morphologyEx(mask_edge_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (5,5)), iterations=3)

cv.namedWindow("mm", cv.WINDOW_NORMAL)
cv.imshow("mm", np.hstack((mask_edge, mask_edge_)))
cv.waitKey(0)
cv.destroyAllWindows()


# cv.namedWindow("Live", cv.WINDOW_NORMAL)
# cv.imshow("Live", mask_edge*255)
# cv.waitKey(0)
# cv.destroyAllWindows()