from PIL import Image, ImageEnhance

# Input Image
img = Image.open('img/MCUCXR_0001_0.png')

# Factor in Contrast & Brightness
ct = 2.0
br = 2.0

# Processing with Contrast
enhancer_ct = ImageEnhance.Contrast(img)
enhancer_ct.enhance(ct).save('result/new_img_ct.png')

# Processing with Brightness
img_new = Image.open('result/new_img_ct.png')
enhancer_br = ImageEnhance.Brightness(img_new)
enhancer_br.enhance(br).save('result/new_img_br_ct.png')

# Getting certain pixels from Modified Image (RGB)
img_target_path = 'result/new_img_br_ct.png'
img_target = Image.open(img_target_path, 'r')

img_target_rgb = img_target.convert("RGB")

width = img_target_rgb.size[0]
height = img_target_rgb.size[1]

pixel_cnt = 0

# for x in range(0, width):
#     for y in range(0, height):
#         if img_target_rgb.getpixel((x, y)) == (0, 0, 0): # Black (0, 0, 0)
#             pixel_cnt += 1
#             #print("x = " + str(x) + ", y = " + str(y)) 


print("contrast = " + str(ct))
print("brightness = " + str(br))
# print("pixel_cnt = " + str(pixel_cnt))