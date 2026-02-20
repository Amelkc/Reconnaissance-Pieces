# finalement pre traitement qu'on garde pour la detection des pieces
import cv2 

def pretraitement(img):
    """
    Prétraitement HSV V : robuste éclairage/textures.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2] 
    # normalisation locale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_v = clahe.apply(v_channel)
    # réduire bruit avec gaussian blur
    gauss_blur = cv2.GaussianBlur(enhanced_v, (11, 11), 0)
    
    return gauss_blur
