import glob
import os
import SimpleITK as sitk
import pydicom
import cv2
import math

def imwrite_kor(filename, img, params=None): 
    try: 
        ext = os.path.splitext(filename)[1] 
        result, n = cv2.imencode(ext, img, params) 
        if result:
            with open(filename, mode='w+b') as f: 
                n.tofile(f) 
                return True
        else: 
            return False 
    except Exception as e: 
        print(e) 
        return False
def DCM_to_NPY_and_JPG(dicom_path, npy_path, jpg_path):
    sitk_image = sitk.ReadImage(dicom_path)
    
    # VOI LUT with window center = 0 and window width = 200 
    window_center = 0
    window_width = 200
    voi_lut_filter = sitk.IntensityWindowingImageFilter()
    voi_lut_filter.SetWindowMinimum(window_center - 0.5 * window_width)
    voi_lut_filter.SetWindowMaximum(window_center + 0.5 * window_width)
    output_image = voi_lut_filter.Execute(sitk_image)
    img = sitk.GetArrayFromImage(output_image)[0,:,:]
    
    # resampling to pixel spacing (0.35, 0.35)
    ratio=(pydicom.read_file(dicom_path)[0x28, 0x0030].value[0]) / 0.35
    original_height,original_width = img.shape
    img = img.astype('float32')
    # padding
    if ratio<1:
        img_rescaled=cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        rescaled_height,rescaled_with=img_rescaled.shape
        height=original_height-rescaled_height
        width=original_width-rescaled_with
        up,down=math.ceil(height/2),math.floor(height/2)
        left,right=math.ceil(width/2),math.floor(width/2)
        npad=((up,down),(left,right))
        img = np.pad(img_rescaled, npad,'constant', constant_values=(0))
    # cropping
    elif ratio>1:
        img_rescaled=cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        rescaled_height,rescaled_with=img_rescaled.shape
        height=rescaled_height-original_height
        width=rescaled_with-original_width
        up,down=math.ceil(height/2),math.floor(height/2)
        left,right=math.ceil(width/2),math.floor(width/2)
        img = img_rescaled[up:rescaled_height-down, left:rescaled_with-right]
    
    # normalization and save
    img = img/255   
    np.save(npy_path,img.astype(np.float32))
    img = img*255
    img = img.astype('uint8')
    imwrite_kor(jpg_path,img)
