import numpy as np
from PIL import Image
import pywt
import pywt.data
import cv2
import math
from zigzag import *

img = cv2.imread("dc.bmp", 0)
vertical=img.shape[0]
horizontal=img.shape[1]
print(type(img))
z_array = np.zeros((vertical,horizontal),dtype=int)
#print(z_array)
e_array = np.zeros((vertical,horizontal),dtype=int)
######  LPC Predictive Coding  ###############
for i in range(0,vertical):
    for j in range(0,horizontal):
        if i==0 and j==0 : #1
            z_array[i][j] = img[i][j] - (img[i][j+1] + img[i+1][j])
          
        elif i==vertical-1 and j==0 : #2
            z_array[i][j] = img[i][j] - (img[i-1][j] + img[i][j+1])
          
        elif i==0 and j==horizontal-1: #3
            z_array[i][j] = img[i][j] - (img[i][j-1] + img[i+1][j])
          
        elif i==vertical-1 and j==horizontal-1: #4
            z_array[i][j] = img[i][j] - (img[i-1][j] + img[i][j-1])
          
        elif i==0 and (j!=horizontal-1 and j!=0): #5
            z_array[i][j] = img[i][j] - (img[i][j-1] + img[i+1][j])
          
        elif i==vertical-1 and (j!=horizontal-1 and j!=0): #6
            z_array[i][j] = img[i][j] - (img[i][j-1] + img[i-1][j])
        elif (i!=0 and i!=vertical-1) and (j!=0 and j!=horizontal-1): #7
            z_array[i][j] = img[i][j] - (img[i][j-1] + img[i-1][j])
        elif (i!=0 and i!=vertical-1) and j==0: #8
            z_array[i][j] = img[i][j] - (img[i][j+1] + img[i-1][j])
        elif (i!=0 and i!=vertical-1) and j==horizontal-1:  #9
            #print(9)
            z_array[i][j] = img[i][j] - (img[i][j-1] + img[i-1][j])
print(z_array)
e_array = img - z_array
print(e_array)
new_image = Image.fromarray(e_array.astype(np.uint8))
new_image.save('after_LPC.bmp')
######  LPC Predictive Coding  ###############3

################# Discrete Wavelet Transform ######################
dwt=pywt.dwt2(e_array,'haar', mode='periodization')
cA, (cH,cV,cD)=dwt
x=cA+cH+cV+cD
#print(x)
new_image = Image.fromarray(x.astype(np.uint8))
new_image.save('after_DWT.bmp')
################### Discrete Wavelet Transform ###################33



###################### Huffman Coding  #######################
block_size = 8
img = cv2.imread('after_DWT.bmp', 0)
cv2.imshow('input image', img)
[h , w] = img.shape
h = np.int32(h)
w = np.int32(w)
nbh = math.ceil(h/block_size)
nbh = np.int32(nbh)
nbw = math.ceil(w/block_size)
nbw = np.int32(nbw)
H =  block_size * nbh
W =  block_size * nbw
padded_img = np.zeros((H,W))
for i in range(h):
        for j in range(w):
                pixel = img[i,j]
                padded_img[i,j] = pixel

cv2.imshow('input padded image', np.uint8(padded_img))
for i in range(nbh):
        row_ind_1 = i*block_size
        row_ind_2 = row_ind_1+block_size
        
        for j in range(nbw):
            col_ind_1 = j*block_size
            col_ind_2 = col_ind_1+block_size
            block = padded_img[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]
            DCT = cv2.dct(block)
            reordered = zigzag(DCT)
            reshaped= np.reshape(reordered, (block_size, block_size))
            padded_img[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshaped

cv2.imshow('encoded image', np.uint8(padded_img))
np.savetxt('encoded.txt',padded_img)
np.savetxt('size.txt',[h, w, block_size])  
##################### Huffman Coding #####################



##################### Huffman DeCoding #####################
padded_img = np.loadtxt('encoded.txt')

[h, w, block_size] = np.loadtxt('size.txt')

[H, W] = padded_img.shape

nbh = math.ceil(h/block_size)
nbh = np.int32(nbh)

nbw = math.ceil(w/block_size)
nbw = np.int32(nbw)


for i in range(nbh):
        row_ind_1 = i*int(block_size)
        row_ind_2 = row_ind_1+int(block_size)

        for j in range(nbw):
            col_ind_1 = j*int(block_size)
            col_ind_2 = col_ind_1+int(block_size)
            block = padded_img[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]
            reshaped= np.reshape(block,(int(block_size)*int(block_size)))
            reordered = inverse_zigzag(reshaped, int(block_size), int(block_size))
            IDCT = cv2.idct(reordered)
            padded_img[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ] = IDCT


padded_img = np.uint8(padded_img)

decoded_img = padded_img[0:int(h),0:int(w)]

new_image = Image.fromarray(decoded_img.astype(np.uint8))
new_image.save('huffman_decode.bmp')

##################### Huffman DeCoding #####################


###################### IDWT ########################
img1 = cv2.imread("after_DWT.bmp", 0)
idwt=pywt.idwt2(dwt,'haar', mode='periodization')
new_image = Image.fromarray(idwt.astype(np.uint8))
new_image.save('after_IDWT.bmp')

###################### IDWT ########################




##################### ILPC  ##########################3
e_array = cv2.imread("after_IDWT.bmp",0)
e_array = np.delete(e_array,553,0)
e_array = np.delete(e_array,537,1)
#x_array = e_array + z_array
print(e_array.shape)
x_array = e_array + z_array
new_image = Image.fromarray(x_array.astype(np.uint8))
new_image.save('original_image.bmp')
###################### ILPC #####################3