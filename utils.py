#fonctions utiles pour éviter redondance code et faciliter les tests
import matplotlib.pyplot as plt
import cv2 as cv

def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def dessin_cercle(img,num_labels, stats, centroids, aire_min):
    #pour debugger mais reverif si cette fonction correcte
    img_debug = img.copy()
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area < aire_min:
            continue
        x, y, w, h, _ = stats[i]
        cx, cy = centroids[i]
        # rayon approximatif
        r = int(0.5 * (w + h) / 2)
        cv.circle(img_debug, (int(cx), int(cy)), r, (0, 0, 255), 2)
