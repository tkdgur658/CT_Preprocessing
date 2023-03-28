import SimpleITK as sitk
import pydicom
import numpy as np
import os
import natsort

# 1) DICOM 파일을 불러와서 3D 객체로 concatenation
def load_dicom_as_3d(folder_path, ref_filename, origin, num_adjacent_files=15):
    
    # get the list of all DICOM files in the input directory
    all_files = natsort.natsorted(os.listdir(folder_path))

    # get the index of the reference file in the list of all files
    ref_file_index = all_files.index(ref_filename)

    # get the indices of the adjacent files
    start_index = max(0, ref_file_index - num_adjacent_files)
    end_index = min(len(all_files) - 1, ref_file_index + num_adjacent_files)

    # create a list of the filenames to process
    filenames = [all_files[i] for i in range(start_index, end_index + 1) if all_files[i].endswith(".dcm")]
    images = []
    for file in filenames:
        image = sitk.ReadImage(os.path.join(folder_path, file))
        image.SetOrigin(origin)
        # dimension reduction if dimension of stack is higher than 2 because it occus error on JoinSeries
        if len(sitk.GetArrayFromImage(image).shape)>=3:
            slice_index = 0 # The index of the slice you want to extract
            direction_index = len(sitk.GetArrayFromImage(image).shape)-1-np.argmin(sitk.GetArrayFromImage(image).shape) # The index of the dimension you want to remove
            image = sitk.Extract(image, [image.GetSize()[i] if i != direction_index else 0 for i in range(image.GetDimension())], [0]*image.GetDimension())
        images.append(image)        
    images = sitk.JoinSeries(images)   
    return images

# 2) resampling
def resample(image, original_spacing, new_spacing):
    original_size = image.GetSize()
    new_size = [int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
    int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
    int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))]
    
    image.SetSpacing(original_spacing)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(image)

# 3) VOI LUT 연산
def apply_voi_lut(image, window_level, window_width):
    windowMinimum = window_level - window_width/2.0
    windowMaximum = window_level + window_width/2.0
    # apply VOI LUT
    voi_lut_filter = sitk.IntensityWindowingImageFilter()
    voi_lut_filter.SetOutputMinimum(0)
    voi_lut_filter.SetOutputMaximum(255)
    voi_lut_filter.SetWindowMaximum(windowMaximum)
    voi_lut_filter.SetWindowMinimum(windowMinimum)

    # VOI LUT 연산 적용
    rescaled_image = voi_lut_filter.Execute(image)
   
    return sitk.Cast(rescaled_image, sitk.sitkUInt8)

def center_crop_or_pad_sitk(img, output_size):
    """
    SimpleITK 이미지를 center crop하거나 zero-padding하여 원하는 크기로 조정합니다.
    
    Args:
        img (SimpleITK.Image): 3D 이미지
        output_size (tuple): 조정할 크기 (height, width, depth)
    
    Returns:
        SimpleITK.Image: 조정된 이미지
    """
    # SimpleITK 이미지를 numpy 배열로 변환
    img_array = sitk.GetArrayFromImage(img)
    
    # center crop 또는 zero-padding
    resized_array = center_crop_pad(img_array, output_size)
    
    return resized_array

def center_crop_pad(img, output_size):
    """
    NumPy 배열로 된 3D 이미지를 x, y 축은 center crop하고 z 축은 zero-padding하여 원하는 크기로 조정합니다.
    
    Args:
        img (numpy.ndarray): 3D 이미지
        output_size (tuple): 조정할 크기 (height, width, depth)
    
    Returns:
        numpy.ndarray: 조정된 이미지
    """
    img_size = np.array(img.shape[:3])
    output_size = np.array(output_size)
    diff = output_size - img_size
    if np.all(diff == 0):
        return img
    crop_size = np.min([img_size, output_size], axis=0)
    start = (img_size - crop_size) // 2
    end = start + crop_size
    cropped_img = img[start[0]:end[0], start[1]:end[1], :crop_size[2]]
    pad_diff = output_size - crop_size
    pad_width = pad_width = ((int(math.ceil(pad_diff[0]/2)), int(math.floor(pad_diff[0]/2))), (int(math.ceil(pad_diff[1]/2)), int(math.floor(pad_diff[1]/2))), (int(math.ceil(pad_diff[2]/2)), int(math.floor(pad_diff[2]/2))))
    padded_img = np.pad(cropped_img, pad_width, mode='constant', constant_values=0)
    return padded_img

def preprocessing_CT(dicom_folder, ref_filename, output_path, npy_or_jpg=True):
    #folder_path = r"C:\Users\LSH\Desktop\2013_04_11"
    #ref_filename = "10028.dcm"
    
    dicom = pydicom.dcmread(f'{dicom_folder}/{ref_filename}')
    slice_thickness = dicom.SliceThickness
    pixel_spacing = dicom.PixelSpacing
    origin = sitk.ReadImage(os.path.join(dicom_folder, ref_filename)).GetOrigin()
    
    # ref 파일을 중심으로 위 아래 16mm 정도만 가져옴.
    num_adjacent_files= int(16/slice_thickness)
        
    # 1) 다수의 (512, 512) 크기의 DICOM 파일을 불러와서 3D 객체로 concatenation
    image = load_dicom_as_3d(dicom_folder, ref_filename, origin, num_adjacent_files=num_adjacent_files)

    # 2) 3D 객체를 x, y, z 축으로 resampling
    image = resample(image, original_spacing = (*pixel_spacing, slice_thickness), new_spacing=(0.35, 0.35, 1))

    # 3) VOI LUT 연산
    image = apply_voi_lut(image, window_level = 0.0, window_width = 200.0)
    
    # 4) crop and padding
    image = center_crop_or_pad_sitk(image, (32, 512, 512))
    # save as npy
    if npy_or_jpg == True :
        image = (image - image.min())/image.max()
        np.save(output_path, image)
    #save as jpg
    else:
        # numpy 배열을 SimpleITK 이미지로 변환
        image = sitk.GetImageFromArray(image)
        # 각 슬라이스를 반복하며 2D 이미지를 저장
        for i in range(image.GetSize()[2]):

            an_image = image[:,:,i]
            # convert the image to a numpy array
            arr = sitk.GetArrayFromImage(an_image)

            # normalize the array to the range [0, 255]
            arr = arr.astype('float32')
            if arr.max() == 0:
                arr= np.zeros(arr.shape)
            else:
                arr = 255*(arr- arr.min())/arr.max()
            arr = arr.astype('uint8')
            
            # save the array as a JPEG image
            sitk.WriteImage(sitk.GetImageFromArray(arr), output_path.replace('.jpg',f'_{i+1}.jpg'))
