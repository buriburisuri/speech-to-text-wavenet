## 0.0.0.2 ( 2017-03-20 )

Features :
    - add more datasets ( VCTK + LibriSpeech, TEDLIUM release 2). 
      Total # of sentence is now 240,612. ( Previously 36,395 )
    - split dataset into train, valid and test to check validation/test loss.
    - add simple speed variation augmentation
    - support multiple GPU training 
    - support Docker image
    
Refactored :
    - adapted to tensorflow 1.0.0
