# Shift-var-deblur

### Shift Variant Image Deconvolution using Deep Learning

Caution: This is a repository which is still in Development stage so there is a possibility that the code might break 
--
Shift Variant Image Deconvolution using Deep Learning We can generate the PSF and the cooresponding patch windows from running the 
[generate_psf.py](scripts/generate_psf.py) script from the scripts folder.
--
We can generate the image pair by using the [generate_Dataset.ipynb](generate_Dataset.ipynb) 
--
Set the '''root_dir''' to the high resolution image dataset. Create two directories to store your blurred and cropped original image for each train dataset as well as validation and test dataset. Keep on changing ```root_dir = <original folder name to store blurred images> ``` ```orig_dir = <original folder name to store cropped originals >``` for all the image folders that you have generated.\
Run for your train and val dataset sequentially.
--
Use the coordConfig.py to set all the arguments for training the network as well as testing it. Run this file
To train the model
```py
python new_train.py
```
To test the model
```py
python test.py
```
