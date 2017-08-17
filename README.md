# Deep Clustering with Convolutional Autoencoders (DCEC)

Keras implementation for ICONIP-2017 paper:

* Xifeng Guo, Xinwang Liu, En Zhu, Jianping Yin. 
Deep Clustering with Convolutional Autoencoders. ICONIP 2017.

## Usage
1. Install [Keras >=v2.0](https://github.com/fchollet/keras), scikit-learn and git   
`sudo pip install keras scikit-learn`   
`sudo apt-get install git`
2. Clone the code to local.   
`git clone https://github.com/XifengGuo/DCEC.git DCEC`
3. Prepare datasets.    

        cd DCEC/data/usps   
        bash ./download_usps.sh   
        cd ../..

4. Run experiment on MNIST.   
`python DCEC.py mnist`     
The DCEC model will be saved to "results/temp/dcec_model_final.h5".

5. Run experiment on USPS.   
`python DCEC.py usps`   

6. Run experiment on MNIST-TEST.   
`python DCEC.py mnist-test`   

