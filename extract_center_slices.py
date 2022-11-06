import os
import numpy as np
import nibabel as nib

def nifti2np(file_path):
    file = nib.load(file_path)
    file = file.get_fdata()
    file = (file-file.min()) / (file.max()-file.min())
    
    if file.ndim != 3:
        print('This data no dims...', file_path.split('/')[-1])
        return None
    
    return file

def origin_nifti2np(file_path):
    file = nib.load(file_path)
    file = file.get_fdata()
    
    if file.ndim != 3:
        print('This data no dims...', file_path.split('/')[-1])
        return None
    
    return file

def set_size_180(data):
    if data.shape[0] != 180:
        zero = 180-data.shape[0]
        data = np.concatenate([np.zeros((zero,180)),data],axis=0)
    if data.shape[1] != 180:
        zero = 180-data.shape[1]
        data = np.concatenate([np.zeros((180,zero)),data],axis=1)
    return data


def extract_center_slices(brain,seg, xmargin=0, ymargin=0, zmargin=0):
    roi = np.where((seg==1) | (seg==4)) # first and second ROI (necrotic + non-enhancing + enhancing)
    
    
    xcenter = int((roi[0].min() + roi[0].max()) / 2)+1+xmargin
    ycenter = int((roi[1].min() + roi[1].max()) / 2)+1+ymargin
    zcenter = int((roi[2].min() + roi[2].max()) / 2)+1+zmargin
    
    sagittal = np.rot90(brain[xcenter,:,:],k=1,axes=(0,1))
    sagittal = np.concatenate([np.zeros((43,240)),sagittal,np.zeros((42,240))],axis=0)
    
    coronal = np.rot90(brain[:,ycenter,:],k=1,axes=(0,1))
    coronal =np.concatenate([np.zeros((43,240)),coronal,np.zeros((42,240))],axis=0)
    
    axial = np.flip(np.rot90(brain[:,:,zcenter],k=3,axes=(0,1)), axis=1)
    
    return axial, coronal, sagittal


def main():
    DIR = "/workspace/Dataset/BRATS2020/"
    MRIs = os.listdir(DIR)[:-2]

    print("Total Num:", len(MRIs))

    total_dataset = 0

    for i in range(len(MRIs)):
    # for i in range(5):
        
        try:

            T1_path = DIR+MRIs[i]+'/'+MRIs[i]+'_t1.nii'
            T1CE_path = DIR+MRIs[i]+'/'+MRIs[i]+'_t1ce.nii'
            T2_path = DIR+MRIs[i]+'/'+MRIs[i]+'_t2.nii'
            FLAIR_path = DIR+MRIs[i]+'/'+MRIs[i]+'_flair.nii'
            SEG_path = DIR+MRIs[i]+'/'+MRIs[i]+'_seg.nii'

            T1 = nifti2np(T1_path)
            T1CE = nifti2np(T1CE_path)
            T2 = nifti2np(T2_path)
            FLAIR = nifti2np(FLAIR_path)
            SEG = origin_nifti2np(SEG_path)

            margins = [[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]

            dataset = 0

            for m in margins:
                x, y, z = m

                t1_axial, t1_coronal, t1_saggital = extract_center_slices(T1, SEG, x, y, z)
                t1ce_axial, t1ce_coronal, t1ce_saggital = extract_center_slices(T1CE, SEG, x, y, z)
                t2_axial, t2_coronal, t2_saggital = extract_center_slices(T2, SEG, x, y, z)
                flair_axial, flair_coronal, flair_saggital = extract_center_slices(FLAIR, SEG, x, y, z)

                axials = np.stack([t1_axial,t1ce_axial,t2_axial,flair_axial],axis=2)
                coronals = np.stack([t1_coronal,t1ce_coronal,t2_coronal,flair_coronal],axis=2)
                saggitals = np.stack([t1_saggital,t1ce_saggital,t2_saggital,flair_saggital],axis=2)

                data = np.stack([axials,coronals,saggitals],axis=3)

                if data.shape != (240,240,4,3):
                    print('This patient failed...', MRIs[i])
                    continue

                data = data[:,:,:,:,np.newaxis]

                if type(dataset) == int:
                    dataset = data
                else:
                    dataset = np.concatenate((dataset,data),axis=4)

            dataset = dataset[:,:,:,:,:,np.newaxis]

            if type(total_dataset) == int:
                total_dataset = dataset
            else:
                total_dataset = np.concatenate((total_dataset,dataset),axis=5)

            print('Success...', MRIs[i], '('+str(i)+'/'+str(len(MRIs))+')')
        
        except:
            print("Fatal Error... pat", MRIs[i])
            continue
            
    if not os.path.isdir("dataset"):
        os.mkdir("dataset")
        
    total_dataset = np.transpose(total_dataset,[5,4,0,1,2,3])
    np.save("dataset/BRATS2020_1(369,7,240,240,4,3)", total_dataset)

    print()
    print('Finished...')
    print('Total patients:', len(MRIs))
    print('Total Dataset shape:',total_dataset.shape)

if __name__ == "__main__":
    main()