# On Disentangling Spoof Traces for Generic Face Anti-Spoofing
Yaojie Liu, Joel Stehouwer, Xiaoming Liu

![alt text](https://yaojieliu.github.io/images/eccv20.png)

## Setup
Tested on Python 3.6 & Tensorflow 1.13.0. As it uses the contrib package, the code should work with Tensorflow >1.8.0 and <1.13.0. The code should be easy to transfer to keras package.

## Training
To run the training code:

    python train.py

The face-anti-spoofing databases (e.g. SiW-M, SiW, and Oulu-NPU) have to be applied separately. We provide the required data format and structure in ./data/ folder. The video should pre-processed into frames of cropped face, saved in one folder under either live/ or spoof/. The landmark (68) should be provided in the XXX.npy file.

## Testing
To run the testing code:

    python test.py

It saves the scores in ./log/XXX/test/score.txt file.

## Acknowledge
Please cite the paper:

    @inproceedings{eccv20yaojie,
        title={On Disentangling Spoof Traces for Generic Face Anti-Spoofing},
        author={Yaojie Liu, Joel Stehouwer, Xiaoming Liu},
        booktitle={In Proceeding of European Conference on Computer Vision (ECCV 2020)},
        address={Virtual},
        year={2020}
    }
    
If you have any question, please contact: [Yaojie Liu](liuyaoj1@msu.edu) 
   
