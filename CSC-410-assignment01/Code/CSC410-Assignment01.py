# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 21:18:07 2020
@author: s_suthah
Modified on Sunday Sept 5th 10:58 PM 2021

"""

import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import numpy as np 
  
# Read Images (e.g, .jpg, .png, and .tif)
cherr_color = cv2.imread("/Users/gilbertec.fleurisma/Downloads/FIDS30/cherries/1.jpg") 
mango_color = cv2.imread("/Users/gilbertec.fleurisma/Downloads/FIDS30/mangos/2.jpg") 
pine_color = cv2.imread("/Users/gilbertec.fleurisma/Downloads/FIDS30/pineapples/40.jpg")

 

# Display the color channels
plt.imshow(cherr_color[:,:,0])
plt.imshow(cherr_color[:,:,1])
plt.imshow(cherr_color[:,:,2])

plt.imshow(mango_color[:,:,0])
plt.imshow(mango_color[:,:,1])
plt.imshow(mango_color[:,:,2])

plt.imshow(pine_color[:,:,0])
plt.imshow(pine_color[:,:,1])
plt.imshow(pine_color[:,:,2])


# Convert to grayscale. 
cherrG = cv2.cvtColor(cherr_color, cv2.COLOR_BGR2GRAY) 
heightcG, widthcG = cherrG.shape 

mangoG = cv2.cvtColor(cherr_color, cv2.COLOR_BGR2GRAY) 
heightmG, widthmG = mangoG.shape 

pineG = cv2.cvtColor(pine_color, cv2.COLOR_BGR2GRAY) 
heightpG, widthpG = pineG.shape 


# Raw data (image) resizing
cherr = cv2.resize(cherrG, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)
mango = cv2.resize(mangoG, dsize=(352, 264), interpolation=cv2.INTER_CUBIC)
pine = cv2.resize(pineG, dsize=(352, 264), interpolation=cv2.INTER_CUBIC)

cherr = cv2.normalize(cherr.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
mango = cv2.normalize(mango.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
pine = cv2.normalize(pine.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255


heightc, widthc = cherr.shape
heightm, widthm = mango.shape
heightp, widthp = pine.shape

# Plot the images
plt.imshow(cherr, cmap=plt.get_cmap('gray'))
plt.axis('off')

plt.imshow(mango, cmap=plt.get_cmap('gray'))
plt.axis('off')

plt.imshow(pine, cmap=plt.get_cmap('gray'))
plt.axis('off')


# Save the images to png/tif
cv2.imwrite('/Users/gilbertec.fleurisma/cherr_grayTD.png', cherr)
cv2.imwrite('/Users/gilbertec.fleurisma/mango_grayTD.png', mango)
cv2.imwrite('/Users/gilbertec.fleurisma/pine_grayTD.png', pine)


#############################################################################

# Some statistical information
print("Statistics", cherr.min(), cherr.max(), round(cherr.mean(),3), round(cherr.std(),4))
print("Statistics", mango.min(), mango.max(), round(mango.mean(),3), round(mango.std(),4))
print("Statistics", pine.min(), pine.max(), round(pine.mean(),3), round(pine.std(),4))



# Binarize an image using a threshold
tmpc = np.zeros((heightc, widthc), np.uint8)
th1 = cherr.mean()
for i in range(heightc):
    for j in range(widthc):
        if(cherr[i][j]<th1):
            tmpc[i][j] = 0
        else:
            tmpc[i][j] = 255  

plt.imshow(tmpc, cmap=plt.get_cmap('gray'))

tmpm= np.zeros((heightm, widthm), np.uint8)
th2 = mango.mean()
for i in range(heightm):
    for j in range(widthm):
        if(mango[i][j]<th2):
            tmpm[i][j] = 0
        else:
            tmpm[i][j] = 255  

plt.imshow(tmpm, cmap=plt.get_cmap('gray'))

tmpp = np.zeros((heightp, widthp), np.uint8)
th3 = pine.mean()
for i in range(heightp):
    for j in range(widthp):
        if(pine[i][j]<th3):
            tmpp[i][j] = 0
        else:
            tmpp[i][j] = 255  

plt.imshow(tmpp, cmap=plt.get_cmap('gray'))


#############################################################################

cc = round(((heightc)*(widthc)))
flatcc = np.zeros((cc, 65), np.uint8)
k = 0
for i in range(heightc-7):
    for j in range(widthc-7):
        crop_tmp = cherr[i:i+8,j:j+8]
        flatcc[k,0:64] = crop_tmp.flatten()
        k = k + 1

fspaceCC = pd.DataFrame(flatcc)  #panda object
fspaceCC.to_csv('/Users/gilbertec.fleurisma/fspaceCC.csv', index=False)


mm = round(((heightm)*(widthm)))
flatmm = np.ones((mm, 65), np.uint8)
k = 0
for i in range(heightm-7):
    for j in range(widthm-7):
        crop_tmp = mango[i:i+8,j:j+8]
        flatmm[k,0:64] = crop_tmp.flatten()
        k = k + 1
        
fspaceMM = pd.DataFrame(flatmm)  #panda object
fspaceMM.to_csv('/Users/gilbertec.fleurisma/fspaceMM.csv', index=False)

pp = round(((heightp)*(widthp)))
flatpp = np.full((pp, 65),2)
k = 0
for i in range(heightc-7):
    for j in range(widthc-7):
        crop_tmp = pine[i:i+8,j:j+8]
        flatpp[k,0:64] = crop_tmp.flatten()
        k = k + 1

fspacePP = pd.DataFrame(flatpp)  #panda object
fspacePP.to_csv('/Users/gilbertec.fleurisma/fspacePP.csv', index=False)


#############################################################################

# Create feature vectors and labels - 0 for cherr and 1 maogo and 2 for pine
cc = round(((heightc)*(widthc))/64)
flatc = np.zeros((cc, 65), np.uint8)
k = 0
for i in range(0,heightc,8):
    for j in range(0,widthc,8):
        crop_tmp1 = cherr[i:i+8,j:j+8]
        flatc[k,0:64] = crop_tmp1.flatten()
        k = k + 1

fspaceC = pd.DataFrame(flatc)  #panda object
fspaceC.to_csv('/Users/gilbertec.fleurisma/fspaceCC.csv', index=False)


mm = round(((heightm)*(widthm))/64)
flatm = np.ones((mm, 65), np.uint8)
k = 0
for i in range(0,heightm,8):
    for j in range(0,widthm,8):
        crop_tmp2 = mango[i:i+8,j:j+8]
        flatm[k,0:64] = crop_tmp2.flatten()
        k = k + 1

fspaceM = pd.DataFrame(flatm)  #panda object
fspaceM.to_csv('/Users/gilbertec.fleurisma/fspaceMM.csv', index=False)


pp = round(((heightp)*(widthp))/64)
flatp = np.ones((pp, 65), np.uint8)
k = 0
for i in range(0,heightp,8):
    for j in range(0,widthp,8):
        crop_tmp3 = pine[i:i+8,j:j+8]
        flatp[k,0:64] = crop_tmp3.flatten()
        k = k + 1

fspaceP = pd.DataFrame(flatp)  #panda object
fspaceP.to_csv('/Users/gilbertec.fleurisma/fspacePP.csv', index=False)


#############################################################################

# Join the feature vectors of the classes
image01 = [fspaceC, fspaceM]
mged01 = pd.concat(image01)

image012 = [fspaceC, fspaceM, fspaceP]
mged012 = pd.concat(image012)

indx1 = np.arange(len(mged01))
rndmged01 = np.random.permutation(indx1)

indx2 = np.arange(len(mged012))
rndmged012 = np.random.permutation(indx2)


rndmged01=mged01.sample(frac=1).reset_index(drop=True)
rndmged012=mged012.sample(frac=1).reset_index(drop=True)


rndmged01.to_csv('/Users/gilbertec.fleurisma/merged01.csv', index=False)
rndmged012.to_csv('/Users/gilbertec.fleurisma/merged012.csv', index=False)