# Binary Classifier

For my CS 315 Data Mining class, we were asked to complete the following assignment:

### Fortune Cookie Classifier

Build a binary fortune cookie classifier. The classifier should sort fortune cookie messages into two classes: messages that predict what will happen in the future and messages that just contain a wise saying. 

For example:

* “Never go in against a Sicilian when death is on the line” would be a wise saying.

* “You will get an A in Machine learning class” would be a message about the future.

<br>

There are three sets of files provided. All words in these files are lower case and punctuation has been removed.

1. The training data:
   traindata.txt: This is the training data consisting of fortune cookie messages.
   trainlabels.txt: This file contains the class labels for the training data.

2. The testing data:
   testdata.txt: This is the testing data consisting of fortune cookie messages.
   testlabels.txt: This file contains the class labels for the testing data.

3. A list of stopwords: stoplist.txt

<br>

There are two steps: the pre-processing step and the classification step. In the pre- processing step, you will convert fortune cookie messages into features to be used by your classifier. You will be using a bag of words representation. The following steps outline the process involved:

1. Form the vocabulary. Let M be the size of your vocabulary.

2. Convert the training data into a set of features. 

3. For each fortune cookie message, you will convert it into a feature vector of size M. Each slot in that feature vector takes the value of 0 or 1. For these M slots, if the ith slot is 1, it means that the ith word in the vocabulary is present in the fortune cookie message; otherwise, if it is 0, then the ith word is not present in the message. Most of these feature vector slots will be 0. 

4. Implement a binary classifier with perceptron weight update as shown below. Use learning rate η=1.

Algorithm 1 Online Binary-Classifier Learning Algorithm

Input: D = Training examples, T = maximum number of training iterations Output: w, the final weight vector

```
Initialize the weights w = 0
for each training iteration itr ∈ {1,2,···,T} do
   for each training example (xt, yt) ∈ D do
      yˆt = sign(w · xt) // predict using the current weights
      if mistake then
         w=w+η·yt·xt //updatetheweights end if
```
      
return final weight vector w


a) Compute the the number of mistakes made during each iteration (1 to 20).

b) Compute the training accuracy and testing accuracy after each iteration (1 to 20).

c) Compute the training accuracy and testing accuracy after 20 iterations with standard perceptron and averaged perceptron.

### OCR Classifier

Given an image of handwritten character, we need to predict whether the corresponding letter is vowel (a, e, i, o, u) or consonant. You are provided with a training and testing set.

Data format. Each non-empty line corresponds to one input-output pair. 128 binary values after “im” correspond to the input features (pixel values of a binary image). The letter immediately afterwards is the corresponding output label.

a) Compute the the number of mistakes made during each iteration (1 to 20).

b) Compute the training accuracy and testing accuracy after each iteration (1 to 20).

c) Compute the training accuracy and testing accuracy after 20 iterations with stanard perceptron and averaged perceptron.


## My Solution

To see my solution, check out the hw.py file and check out the output.txt file for the results of the program. Or, you can try running the program for yourself by downloading Python 3 yoursef and running `python3 hw.py` in terminal
