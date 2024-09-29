from PIL import Image
import numpy as np

def rgb_to_yuv(r, g, b):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.169 * r - 0.331 * g + 0.5 * b + 128
    v = 0.5 * r - 0.419 * g - 0.081 * b + 128
    return y, u, v

def rgb_to_ycbcr(r, g, b):
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    cb = -0.148 * r - 0.291 * g + 0.439 * b + 128
    cr = 0.439 * r - 0.368 * g - 0.071 * b + 128
    return y, cb, cr

def process_image(image_path):
    img = Image.open(image_path)
    img_rgb = img.convert('RGB')
    width, height = img_rgb.size
    
    r_array = np.zeros((height, width), dtype=np.uint8)
    g_array = np.zeros((height, width), dtype=np.uint8)
    b_array = np.zeros((height, width), dtype=np.uint8)
    y_array = np.zeros((height, width), dtype=np.uint8)
    u_array = np.zeros((height, width), dtype=np.uint8)
    v_array = np.zeros((height, width), dtype=np.uint8)
    cb_array = np.zeros((height, width), dtype=np.uint8)
    cr_array = np.zeros((height, width), dtype=np.uint8)
    
    for x in range(width):
        for y in range(height):
            r, g, b = img_rgb.getpixel((x, y))
            
            r_array[y, x] = r
            g_array[y, x] = g
            b_array[y, x] = b
            
            y_yuv, u, v = rgb_to_yuv(r, g, b)
            y_array[y, x] = int(y_yuv)
            u_array[y, x] = int(u)
            v_array[y, x] = int(v)
            
            y_ycbcr, cb, cr = rgb_to_ycbcr(r, g, b)
            cb_array[y, x] = int(cb)
            cr_array[y, x] = int(cr)
    
    Image.fromarray(r_array).save('r_channel.png')
    Image.fromarray(g_array).save('g_channel.png')
    Image.fromarray(b_array).save('b_channel.png')
    Image.fromarray(y_array).save('y_channel.png')
    Image.fromarray(u_array).save('u_channel.png')
    Image.fromarray(v_array).save('v_channel.png')
    Image.fromarray(cb_array).save('cb_channel.png')
    Image.fromarray(cr_array).save('cr_channel.png')

if __name__ == '__main__':
    process_image('./lena.png')