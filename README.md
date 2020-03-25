# Score-based Face Quality Assessment (FQA)
Implementation of [this paper](https://www.researchgate.net/publication/327530639_Score_based_Face_Quality_Assessment_FQA) in Python - OpenCv 

## Important 
- This implementaion is not complete yet.
- Main file name is fqa_score.py and you can go to the fqa directory to access codes.
- You should download [this](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) file and extract it in `fqa` directory

## FQA score 
This script will train a stacked auto encoders on 4 kind of features:
- GIST (I don't know it's abbreviation, you can google it 😀)
- CNN (Transfer Learning)
- HOG (Histogram of graident)
- LBP(local binary pattern)

You should install gist from [here](https://github.com/tuttieee/lear-gist-python) based on your OS
full instruction are availble in link

It uses lfw Dataset for average quality pics, 
FERRET dataset for good pictures and 
False positive faces from face detection programs (like retina) labeled as bad pics.

## Scenario 
We have 3 classes of pics. good, average and bad pictures are available in 'pics' directory.
 we extact every feature for thease and use feature reduction (auto encoders) to downsize it
 to a vector of 50 elemnts. then concate all of the vectores ( 4 * 50 =200) and use feature 
reduction and auto encoders and a softmax layer for classifying images.
### Sample mages
| Original        | Cropped           | Local Binary pattern  | Histogeram of Oriented Gradients|
| :-------------: |:-------------:| :-----:|:-------------:|
| ![alt text](https://github.com/AminTaheri23/Score-based-Face-Quality-Assessment-FQA/blob/master/fqa/lenna%20-%20Copy.jpg "Original")    | ![alt text](https://github.com/AminTaheri23/Score-based-Face-Quality-Assessment-FQA/blob/master/fqa/img_cropped.jpg "Cropped")| ![alt text](https://github.com/AminTaheri23/Score-based-Face-Quality-Assessment-FQA/blob/master/fqa/lbp.jpg "Local Binary Pattern") |![alt text](https://github.com/AminTaheri23/Score-based-Face-Quality-Assessment-FQA/blob/master/fqa/hog.jpg "Histogram of Oriented Gradients")|

## Performance 
Our bottle neck is feature calculation. we should use a trained auto encoder for
 feature reduction (after we tuned it). 

## Attention 
Need to make 'pics' folder if it's not available 

## TODO
[ ] Fine tune the auto encoders
[ ] Concatinate (late fusion) vectors 
[ ] Train an auto encoder (with softmax) for classification
[ ] try V up architectures for auto encoding
[ ] gather more data for pics ( we already have about 1000 pics per class,
 but with FERRET data set that is a 'good' labled, we can extend our data set) 


## Aknwoledgements				      
Seyed Mohammad Amin Taheri at Shenasa-ai.ir during internship 
for any questions, don't hesitate to open and issue or contact me on: amintaheri90@gmail.com  
