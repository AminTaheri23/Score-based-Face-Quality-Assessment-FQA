# Score-based-Face-Quality-Assessment-FQA
implementation of [this paper](https://www.researchgate.net/publication/327530639_Score_based_Face_Quality_Assessment_FQA) in Python - OpenCv 

## Important 
- this implementaion is not complete yet.

- Main file name is fqa_score.py and you can go to the fqa directory to access codes.

- You should download [this](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) file and extract it in `fqa` directory

## FQA score 
this script will train a stacked auto encoders on 4 kind of features,
 GIST(i dont know its abbreviation, you can google it 😀 ),
 CNN,
 HOG(histogram of graident), LBP(local binary pattern)

you should install gist from [here](https://github.com/tuttieee/lear-gist-python) based on your OS
full instruction are availble in link

it uses lfw for average pics, 
FERRET for good pics and 
False positive faces from face detection programs (like retina) as bad pics.

## Scenario 
we have 3 classes of pics. good, average and bad pics are availabel in 'pics' folder.
 we extact every feature for thease and use feature reduction (auto encoders) to downsize it
 to a vector of 50 elemnts. then concate all of the vectores ( 4 * 50 =200) and use feature 
reduction and auto encoders and a softmax layer for classifying images.
### sample mages
| Original        | Cropped           | Local Binary pattern  | Histogeram of Oriented Gradients|
| :-------------: |:-------------:| :-----:|:-------------:|
| ![alt text](https://github.com/AminTaheri23/Score-based-Face-Quality-Assessment-FQA/blob/master/fqa/lenna%20-%20Copy.jpg "Original")    | ![alt text](https://github.com/AminTaheri23/Score-based-Face-Quality-Assessment-FQA/blob/master/fqa/img_cropped.jpg "Cropped")| ![alt text](https://github.com/AminTaheri23/Score-based-Face-Quality-Assessment-FQA/blob/master/fqa/lbp.jpg "Local Binary Pattern") |![alt text](https://github.com/AminTaheri23/Score-based-Face-Quality-Assessment-FQA/blob/master/fqa/hog.jpg "Histogram of Oriented Gradients")|

## Performance 
our bottle neck is feature calculation. we should use a trained auto encoder for
 feature reduction (after we tuned it). 

## Attention 
Need to make 'pics' folder if it's not available 

## TODO : 
- fine tune the auto encoders
- concatinate (late fusion) vectors 
- train an auto encoder (with softmax) for classification 
- gather more data for pics ( we already have about 1000 pics per class,
 but with FERRET data set that is a 'good' labled, we can extend our data set) 


## Aknwoledgements				      
Seyed Mohammad Amin Taheri at Shenasa-ai.ir during  internship 
for any questions, don't hesitate to open and issue or contact me on: amintaher90@gmail.com  
