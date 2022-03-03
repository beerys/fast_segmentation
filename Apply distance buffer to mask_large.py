#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rasterio
import numpy as np
import cv2


# In[2]:


with rasterio.open("data/masks/16_pre_imagery_cropped_large_mask.tif") as f:
    mask = f.read().squeeze()
    mask_profile = f.profile


# In[3]:


BACKGROUND_CLASS_LABEL = 1
BUILDING_CLASS_LABEL = 3

nodata_mask = (mask != BUILDING_CLASS_LABEL).astype(np.uint8)


# In[4]:


transform = cv2.distanceTransform(
    nodata_mask, distanceType=cv2.DIST_L2, maskSize=3
)


# In[5]:


BUFFER_SIZE = 3 # meters -- this is in units of the CRS
background_mask = (transform > 0) & (transform < BUFFER_SIZE)


# In[6]:


mask[background_mask] = BACKGROUND_CLASS_LABEL 


# In[7]:


with rasterio.open("data/masks/16_pre_imagery_cropped_mask_buffered.tif", "w", **mask_profile) as f:
    f.write(mask, 1)

