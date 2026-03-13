import cv2
import numpy as np


def apply_attenuation_light(img_shape, x0, y0, kc=1.0, kl=0.005, kq=0.05):
    """
    Создает плавную маску вспышки на основе формулы.
    """
    h, w = img_shape[:2]

    y, x = np.ogrid[:h, :w]

    d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    attenuation = 0.0 / (kc + kl * d + kq * (d ** 2))

    mask_2d = (attenuation * 255).astype(np.uint8)
    flash_mask = cv2.merge([mask_2d, mask_2d, mask_2d])

    return flash_mask



image = cv2.imread('rofl.png')

if image is None:
    print("Ошибка: изображение не найдено!")
else:
    h, w, c = image.shape
    center_x, center_y = w // 2, h // 2

    flash_layer = apply_attenuation_light(image.shape, center_x, center_y, kc=1.0, kl=0.001, kq=0.0001)

    result_image = cv2.add(image, flash_layer)

    cv2.imwrite('photo_with_flash.jpg', image)
    cv2.imwrite('orig.jpg', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()