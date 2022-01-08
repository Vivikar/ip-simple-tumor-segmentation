from types import FunctionType

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.metrics import f1_score
from tqdm import tqdm
from scipy import ndimage, misc
from scipy.ndimage.morphology import binary_opening, binary_closing
from skimage.metrics import hausdorff_distance
from medpy.filter.smoothing import anisotropic_diffusion

modality_names = ['t1', 't2', 'flair','t1ce']
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    4: 'ENHANCING'  # original 4 -> converted into 3 later
}



def min_max_normalization(image: np.ndarray):
    brain_volume = image[image != 0]
    non_brain_mask = image.copy() == 0

    img_min = np.percentile(brain_volume, 1)
    img_max = np.percentile(brain_volume, 99)
    image[image > img_max] = img_max
    image[image < img_min] = img_min
    image_scaled = (image - img_min)/(img_max - img_min)

    image_scaled[non_brain_mask] = 0

    return image_scaled


def linear_transform_scaling(image: np.ndarray, q25_map=600, q75_map=800, min_value=200, max_value=1200):
    image = image.copy()
    roi_image = image[image != 0]
    q25 = np.percentile(roi_image, 25)
    q75 = np.percentile(roi_image, 75)

    alpha = (q75_map - q25_map)/(q75-q25)
    beta = q25_map - ((q75_map - q25_map)*q25)/(q75-q25)

    roi_image = alpha*roi_image + beta

    roi_image[roi_image > max_value] = max_value
    roi_image[roi_image < min_value] = min_value

    image[image != 0] = roi_image
    return image


def plot_image_histogram(image):
    ax = plt.hist(image.ravel(), bins=int(image.max()), alpha=0.5)


def preprocess_image(image: np.ndarray, normalization_function: FunctionType):
    """Applies pre-procesing pipeline to a given image

    Args:
        image (np.ndarray): Image (2D or 3D to be processed)
        normalization_function (function): Function to use for data normalization/scaling

    Returns:
        np.ndarray: Pre-processed image (normalization + anisotripic diffusion)
    """

    return anisotropic_diffusion(normalization_function(image))


def retrieve_brats_data(patients, modalities, brats_path='', normalize=True, dataset_part="_Training_"):
    """Retrieves BRATS2020 data for given patients of given MR modalities

    Args:
        patients ([int]): List of patients to extract data from
        modalities (List[str]): Modalities to extract (allowed are ['t1', 't2', 't1ce', 'flair'])
        brats_path (str, optional): Path to brats folder. Defaults to ''.

    Returns: #+1 bcs we are extracting the mask first and then the 4 modalities, x y z is 3d
        np.ndarray: Array with extracted data of shape (n_patients, n_modalities + 1, x_dim, y_dim, z_dim)
        Where x_dim, y_dim, z_dim are dimensions of the 3D image volumes (240, 240, 155) and
        and n_modalities + 1 contains segmentation mask at 0 index and then all extracted modalities in the
        supplied order
    """
    data = []
    
    for patient_idx, patient in enumerate(patients):
        if patient < 10:
            brain_path = f'BraTS20{dataset_part}00{patient}/BraTS20{dataset_part}00{patient}'
            segm_path = f'BraTS20{dataset_part}00{patient}/BraTS20{dataset_part}00{patient}_seg.nii'
        elif 10 <= patient < 100:
            brain_path = f'BraTS20{dataset_part}0{patient}/BraTS20{dataset_part}0{patient}'
            segm_path = f'BraTS20{dataset_part}0{patient}/BraTS20{dataset_part}0{patient}_seg.nii'

        else:
            brain_path = f'BraTS20{dataset_part}{patient}/BraTS20{dataset_part}{patient}'
            segm_path = f'BraTS20{dataset_part}{patient}/BraTS20{dataset_part}{patient}_seg.nii'
        if dataset_part == "_Training_":
            data.append([nib.load(brats_path + segm_path).get_fdata()])
        else:
            data.append([np.zeros((240,240,155))])
        for modality in modalities:
            single_modality_brain_path = brain_path + f'_{modality}.nii'
            if normalize:
                data[patient_idx].append(preprocess_image(nib.load(
                    brats_path + single_modality_brain_path).get_fdata(), min_max_normalization))
            else:
                data[patient_idx].append(
                    nib.load(brats_path + single_modality_brain_path).get_fdata())
    return np.array(data)


def dice_score(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError(
            "Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def variance_filter(image, size):
    win_mean = uniform_filter(image, size=size)
    win_sqr_mean = uniform_filter(image**2, size=size)
    win_var = win_sqr_mean - win_mean**2
    return win_var


def skeweness_filter(img, size):
    imgS = img*img
    m, m2p, m3p = (uniform_filter(x, size) for x in (img, imgS, imgS*img))
    mS = m*m
    skew_img = (m3p-3*m*m2p+2*mS*m)/(m2p-mS)**1.5

    skew_img[img == 0] = 0
    skew_img = min_max_normalization(np.nan_to_num(skew_img))
    return skew_img


def kurtosis_filter(img, size):
    imgS = img*img
    m, m2p, m3p, m4p = (uniform_filter(x, size)
                        for x in (img, imgS, imgS*img, imgS*imgS))
    mS = m*m

    kurt_img = (m4p - 4*m3p*m + 6*m2p*mS - 3*mS*mS)/(m2p-mS)

    kurt_img[img == 0] = 0
    kurt_img = min_max_normalization(np.nan_to_num(kurt_img))
    return kurt_img


def feature_extractor_1(patient_scans, bx_idxs, by_idxs, bz_idxs, kernel_size=3):
    """Extracts voxel-wise features from a given set of brain scans

    Args:
        patient_scans (np.ndarray(5,x_dim,y_dim,z_dim)): Patients MRI scans and segmentation mask in the order
        [segmentation_scan, t1, t2, t1ce, flair]
        kernel_size (int, optional): Defines kernel_size for filters used in FE. Defaults to 3.

    Returns:
        (X, y, (bx_idxs, by_idxs, bz_idxs)): X - extracted features in a matrix of shape (n_voxels, n_features)
        y - true segmentation classes (n_voxels,)
        (bx_idxs, by_idxs, bz_idxs) - coordinates of the voxels features were extracted from (brain voxel coordinates)
    """
    # intensitiesc
    X = np.stack([patient_scans[i][(bx_idxs, by_idxs, bz_idxs)]
                  for i in range(1, len(patient_scans))]).T

    # filters
    mean_f = np.stack((uniform_filter(patient_scans[i], size=kernel_size,)[
        (bx_idxs, by_idxs, bz_idxs)] for i in range(1, len(patient_scans)))).T
    var_f = np.stack((variance_filter(patient_scans[i], size=kernel_size,)[
        (bx_idxs, by_idxs, bz_idxs)] for i in range(1, len(patient_scans)))).T
    skew_f = np.stack((skeweness_filter(patient_scans[i], size=kernel_size,)[
        (bx_idxs, by_idxs, bz_idxs)] for i in range(1, len(patient_scans)))).T
    kurtosis_f = np.stack((kurtosis_filter(patient_scans[i], size=kernel_size,)[
        (bx_idxs, by_idxs, bz_idxs)] for i in range(1, len(patient_scans)))).T

    X = np.hstack([X, mean_f, var_f, skew_f, kurtosis_f])
    y = patient_scans[0][bx_idxs, by_idxs, bz_idxs]
    return X, y


def t1_median(study):
    study_filtered = np.zeros((240,240,155))
    if np.size(np.shape(study)) == 5:
        for k in range(np.shape(study)[-1]):
            #study_filtered[:,:,k] = ndimage.median_filter(study[:,:,k],size=(11,11),mode='constant')
            study_filtered[:,:,k] = ndimage.median_filter(study[0,1,:,:,k],size=(11,11),mode='constant')
    else:
        for k in range(np.shape(study)[-1]):
            #study_filtered[:,:,k] = ndimage.median_filter(study[:,:,k],size=(11,11),mode='constant')
            study_filtered[:,:,k] = ndimage.median_filter(study[0,:,:,k],size=(11,11),mode='constant')
    return study_filtered

def flair_mean(study,kernel_size):
    kernel = np.ones((kernel_size,kernel_size))/(kernel_size**2)
    study_filtered = np.zeros((240,240,155))
    if np.size(np.shape(study)) == 5:
        for k in range(np.shape(study)[-1]):
            #study_filtered[:,:,k] = ndimage.convolve(study[:,:,k],kernel,mode='constant')
            study_filtered[:,:,k] = ndimage.convolve(study[0,3,:,:,k],kernel,mode='constant') 
    else:
        for k in range(np.shape(study)[-1]):
            #study_filtered[:,:,k] = ndimage.convolve(study[:,:,k],kernel,mode='constant')
            study_filtered[:,:,k] = ndimage.convolve(study[2,:,:,k],kernel,mode='constant') 
    return study_filtered

       

def evaluate_brain(patient_scans, feature_extraction_func, model, sample=False, kernel_size=3):
    """Evaluates model performance on tumor classification of a given model

    Args:
        patient_scans (np.ndarray(5,x_dim,y_dim,z_dim)): Patients MRI scans and segmentation mask in the order
        feature_extraction_func (function): Function to extract features for the model.
        Should accept following parameters:
        Should return: 
        model (sklear.model): Model with model.predict() method
        sample (bool, optional): Whether to sample brain for evaluation or nor.
            If False: all brain voxels are used for evaluation and the final predicted mask
                is returned (of shape (x_dim, y_dim, z_dim).
            If int>0:  sampled brain voxels for evaluation for the speed-up.
            Returns array of predictions for each vosel (n_voxels, 1). 
        Defaults to False.
        kernel_size (int, optional): Defines kernel_size for filters used in FE. Defaults to 3.

    Returns:
        [type]: [description]
    """
    brain_mask = (patient_scans[1] != 0) & (
        patient_scans[2] != 0) & (patient_scans[3] != 0)
    bx_idxs, by_idxs, bz_idxs = np.where((brain_mask != 0).astype(bool))

    if sample:
        sampled_tumor_points = np.random.randint(
            low=0, high=len(bx_idxs), size=sample)

        bx_idxs = bx_idxs[sampled_tumor_points]
        by_idxs = by_idxs[sampled_tumor_points]
        bz_idxs = bz_idxs[sampled_tumor_points]

    print("Feature Extraction...")
    X, y = feature_extraction_func(
        patient_scans, bx_idxs, by_idxs, bz_idxs, kernel_size=3)

    print("Prediction...")
    y_predicted = model.predict(X)

    if not sample:
        print("Compiling the predicted mask...")
        predicted_mask = np.zeros(patient_scans[0].shape)
        for b_idx in tqdm(range(len(bx_idxs))):
            predicted_mask[bx_idxs[b_idx], by_idxs[b_idx],
                           bz_idxs[b_idx]] = y_predicted[b_idx]
        return predicted_mask
    else:
        return y_predicted


def calculate_metrics(patient_data, predicted_mask):
    brain_mask = patient_data[0, 1] > 0
    for i in SEGMENT_CLASSES.keys():
        print(f'DSC for {SEGMENT_CLASSES[i]}: ', f1_score(
            (patient_data[0, 0] == i).flatten(), (predicted_mask == i).flatten()))
        print()

    print('DSC for all tumor regions: ', f1_score(
        patient_data[0, 0][brain_mask] > 0, predicted_mask[brain_mask] > 0))


def plot_results(patient_data, predicted_mask, slice,patient_no,dataset_part):
    """Plots results of the segmentation along with 
    patient scans in different modalities

    Args:
        patient_data (np.ndarray(5,x_dim,y_dim,z_dim)): Patients MRI scans and segmentation mask in the order
        predicted_mask (np.ndarray(x_dim, y_dim, z_dim)): Predicted 3D volume of segmentation
        slice (int): Slice number
    """
    fig, ax = plt.subplots(2, 4, figsize=(12, 8))
    fig.suptitle('MRI Scan, patient: ' + str(patient_no) + ' ('+dataset_part+')' + ' , slice: ' + str(slice),fontsize=20)
    fig.tight_layout()
    for mri_mod in range(4):
        ax[0,mri_mod].imshow(patient_data[0, mri_mod + 1]
                              [:, :, slice], cmap='gray')
        ax[0,mri_mod].set_title(modality_names[mri_mod] + f' {slice} slice')
        ax[0,mri_mod].axis('off')

    ax[1,0].imshow(patient_data[0, 0][:, :, slice], cmap='gray')
    ax[1,0].set_title("Original Mask")
    ax[1,0].axis('off')

    ax[1, 1].imshow(predicted_mask[:, :, slice], cmap='gray')
    ax[1, 1].set_title("Predicted Mask")
    ax[1, 1].axis('off')

    ax[1,2].imshow(patient_data[0, 0][:, :, slice] > 0, cmap='gray')
    ax[1,2].imshow(predicted_mask[:, :, slice] > 0, cmap='gray', alpha=0.5)
    ax[1,2].set_title("Predicted over Original Mask")
    ax[1,2].axis('off')

    ax[1,3].imshow(patient_data[0, 3][:, :, slice], cmap='gray')
    ax[1,3].imshow(predicted_mask[:, :, slice] > 0, cmap='gray', alpha=0.5)
    ax[1,3].set_title("Predicted mask over Flair")
    ax[1,3].axis('off')

    plt.tight_layout()
    plt.show()

def post_process(image):
    struct_element = np.ones((3, 3, 3))
    return binary_closing(binary_opening(image, struct_element))

def obtain_test_data(patients,dataset_path,brats_path):
    no_px = np.shape(patients)[0]
    #modalities = ['t1','t2','flair','t1ce']
    #random_sample_size = 2000
    data_set = []
    
    
    for i in range(no_px):
        px_ID = [patients[i]]
        study = retrieve_brats_data(px_ID,modality_names,normalize=False,dataset_part=dataset_path,brats_path=brats_path)
        #Obtain brainmask
        brain_mask = np.where(study[0,1] != 0)
        #Min-max normalization
        for j in range(1,np.shape(study)[1]):
            min_value = study[0,j].min()
            max_value = study[0,j].max()
            study[0,j] = (study[0,j]-min_value)/(max_value-min_value)           
        #Homogenize tumour
        study[0,0] = (study[0,0] != 0).astype(int)
        #Feature extraction
        study_FE = []
        study_FE.append([study[0,0]]) #tumour mask
        #T2 min 3x3x3
        study_FE[0].append(ndimage.minimum_filter(study[0,2,:,:,:],size=(3,3,3),mode='constant'))
        #T1 min 3x3x3
        study_FE[0].append(ndimage.minimum_filter(study[0,1,:,:,:],size=(3,3,3),mode='constant'))
        #FLAIR min 3x3x3
        study_FE[0].append(ndimage.minimum_filter(study[0,3,:,:,:],size=(3,3,3),mode='constant'))
        #T1CE min 3x3x3
        study_FE[0].append(ndimage.minimum_filter(study[0,4,:,:,:],size=(3,3,3),mode='constant'))
        #T2 max 3x3x3
        study_FE[0].append(ndimage.maximum_filter(study[0,2,:,:,:],size=(3,3,3),mode='constant'))
        #Flair max 3x3x3
        study_FE[0].append(ndimage.maximum_filter(study[0,3,:,:,:],size=(3,3,3),mode='constant'))
        #T1 max 3x3x3
        study_FE[0].append(ndimage.maximum_filter(study[0,1,:,:,:],size=(3,3,3),mode='constant'))
        #T1ce max 3x3x3
        study_FE[0].append(ndimage.maximum_filter(study[0,4,:,:,:],size=(3,3,3),mode='constant'))
        #T1 median 11x11x11
        study_FE[0].append(ndimage.median_filter(study[0,1,:,:,:],size=(11,11,11),mode='constant'))
        #T1 median 11x11
        study_FE[0].append(t1_median(study))
        #Flair mean 11x11
        study_FE[0].append(flair_mean(study,11))
        #Flair mean 9x9
        study_FE[0].append(flair_mean(study,9))
            #print('study_FE size: ',np.shape(study_FE))
        #sample data
        no_features = np.shape(study_FE)[1]
        data_set = np.zeros((np.shape(brain_mask)[1],no_features-1)) #create data set of voxels of 1 patient
            #print('data_set size: ',np.shape(data_set))
        for j in range(1,no_features):
            data_set[:,j-1] = study_FE[0][j][brain_mask] #fill each column with reduced number of voxels (only brain)
        
    
    #Get labels of test data_set for evaluation
    true_tumour_mask = study[0,0][brain_mask]
    
    #Get indexes for predicted mask reconstruction
    x_v,y_v,z_v = brain_mask
    vox_idx = []
    for idx_pos in range(x_v.shape[0]):
        vox_idx.append(np.array([x_v[idx_pos],y_v[idx_pos],z_v[idx_pos]]))
    original_mask = study[0,0]
    
    return data_set,true_tumour_mask,vox_idx,original_mask,study


def rf_prediction(patient_number,trained_classifier,dataset_part,brats_path):
    """Given a trained classifier, predicts a tumour mask for a patient

    Args:
        patient_number (list[int]): Patient to predict tumour mask
        trained_classifier: classifier previously trained
        dataset_part: to select either training or validation patient

    Returns:
        f1s_predicted: F1 Score of the brain voxels
        hd_predicted: Hausdorff distance of whole volume
        pred_TM_px1_final: Final volume of predicted mask
        patient_data: original array of selected patient study
    """
    test_arr_2,true_TM_px1_BM,vox_idx,original_mask,patient_data = obtain_test_data(patient_number,dataset_part,brats_path)
    
    pred_TM_px1_BM = trained_classifier.predict(test_arr_2)
    
    f1s_predicted = f1_score(true_TM_px1_BM,pred_TM_px1_BM)
    
    #Predicted volume reconstruction
    pred_TM_px1_final = np.zeros((240,240,155))
    for pred_idx,pred in enumerate(pred_TM_px1_BM):
        pred_TM_px1_final[vox_idx[pred_idx][0],vox_idx[pred_idx][1],vox_idx[pred_idx][2]] = pred
    
    hd_predicted = hausdorff_distance(original_mask,pred_TM_px1_final)

    return f1s_predicted,hd_predicted,pred_TM_px1_final,patient_data
