This folder contains the source code and the datasets we used in the paper. 

Synthetic dataset needs to be generated first locally. By running the script 
python3 sample_generation.py

Please make sure that your machine installed CUDA, pytorch, sklearn, cv2 and other libraries, such as numpy, scipy, matplotlib. 

Each dataset can be run inside the folder, such as

python3 train_cnns.py
python3 train_lcnn.py
python3 train_ae_svm.py
