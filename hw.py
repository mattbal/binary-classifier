# Perceptron Binary Classifier by Matt Balint

import numpy as np

# Part 1. Fortune Cookie Classifier

def read(infile, as_int = False):
  lines = []
  with open(infile) as f:
    lines = f.read().split("\n")

  lines = np.array(lines)
  if as_int:
    return lines.astype(int)
  else:
    return lines

def get_vocabulary(data, stoplist):
  vocabulary = np.array([])
  for fortune in data:
    vocabulary = np.append(vocabulary, fortune.split())

  vocabulary = np.unique(vocabulary)

  indexes_to_delete = []
  for i, word in enumerate(vocabulary):
    for x in stoplist:
      if word == x:
        indexes_to_delete.append(i)
        break

  return np.delete(vocabulary, indexes_to_delete)

def get_features(data, vocabulary):
  features = []
  for fortune in data:
    temp = np.zeros(len(vocabulary))
    for word in fortune.split():
      for i, x in enumerate(vocabulary):
        if word == x:
          temp[i] = 1
          break
    features.append(temp)

  return features



print("--- Fortune Cookie ---\n")

fortune_stop = read('./fortune-cookie-data/stoplist.txt')
fortune_training_data = read('./fortune-cookie-data/traindata.txt')
fortune_training_labels = read('./fortune-cookie-data/trainlabels.txt', True)
fortune_test_data = read('./fortune-cookie-data/testdata.txt')
fortune_test_labels = read('./fortune-cookie-data/testlabels.txt', True)
print("Successfully imported all fortune cookie data.\n")

# Fix the labels. In these files, 0 is incorrectly used instead of -1. If we use 0, the weights will stop adjusting after 2nd iteration and we will have a lot of mistakes
fortune_training_labels[fortune_training_labels == 0] = -1
fortune_test_labels[fortune_test_labels == 0] = -1

# Form the vocabulary. 
# The vocabulary is the set of all the words that are in the training data with stop words removed (stop words are common, uninformative words
# such as “a” and “the” that are listed in the file stoplist.txt)
vocabulary = get_vocabulary(fortune_training_data, fortune_stop)
print("Finished processing vocabulary.\n")

features = np.array(get_features(fortune_training_data, vocabulary))
features_test = np.array(get_features(fortune_test_data, vocabulary)) # Use the same vocabulary. If we don't, we won't be able to test the weight since the weight will be a different size than the test data's features

# Standard Perceptron
print("Running Standard Perceptron...\n")
w = np.zeros(len(vocabulary)) # initialize weight
mistakes_training = []
mistakes_testing = []

for i in range(20):
  mistakes = 0

  for j, x in enumerate(features):
    y = fortune_training_labels[j]
    prediction = np.sign(np.dot(w, x))
    if y * prediction <= 0: # mistake was made
      w = w + y * x
      mistakes += 1

  mistakes_training.append(mistakes)

  # Test the weight on testing data
  mistakes = 0

  for j, x in enumerate(features_test):
    y = fortune_test_labels[j]
    prediction = np.sign(np.dot(w, x))
    if y * prediction <= 0: # mistake was made
      mistakes += 1

  mistakes_testing.append(mistakes)


# Avg Perceptron
print("Running Avg Perceptron...\n")
w = np.zeros(len(vocabulary)) # initialize weight
w_sum = np.zeros(len(vocabulary))
c = 1 # count

for i in range(20):

  for j, x in enumerate(features):
    y = fortune_training_labels[j]
    prediction = np.sign(np.dot(w, x))
    if y * prediction <= 0: # mistake was made
      w = w + y * x
      w_sum = w_sum + c * w
      mistakes += 1
      c += 1

avg_w = w_sum / c

# Calculate the training and testing accuracy
mistakes = 0

for j, x in enumerate(features):
  y = fortune_training_labels[j]
  prediction = np.sign(np.dot(w_sum, x))
  if y * prediction <= 0: # mistake was made
    mistakes += 1

avg_mistakes_training = mistakes

mistakes = 0

for j, x in enumerate(features_test):
  y = fortune_test_labels[j]
  prediction = np.sign(np.dot(w_sum, x))
  if y * prediction <= 0: # mistake was made
    mistakes += 1

avg_mistakes_testing = mistakes



# Output our results
output = open('output.txt', 'w')
output.write("Fortune Cookie\n\n")
output.write("\t\tMistakes\n")
for i, x in enumerate(mistakes_training):
  output.write(f"Iteration-{i+1}\t{x}\n")

output.write("\n")
num_training_examples = len(fortune_training_data)
num_testing_examples = len(fortune_test_data)
output.write("\t\tTraining Accuracy\tTesting Accuracy\n")
for i, x in enumerate(mistakes_training):
  output.write(f"Iteration-{i+1}\t{(num_training_examples - x)/num_training_examples}\t{(num_testing_examples - mistakes_testing[i])/num_testing_examples}\n")

output.write("\n")
output.write("Standard Perceptron Training Accuracy\tAvg Perceptron Training Accuracy\n")
output.write(f"{(num_training_examples - mistakes_training[19])/num_training_examples}\t\t\t{(num_training_examples - avg_mistakes_training)/num_training_examples}\n")
output.write("Standard Perceptron Testing Accuracy\tAvg Perceptron Testing Accuracy\n")
output.write(f"{(num_testing_examples - mistakes_testing[19])/num_testing_examples}\t\t\t{(num_testing_examples - avg_mistakes_testing)/num_testing_examples}\n\n")




# Part 2. OCR Classifier
def read_ocr(infile):
  lines = []
  with open(infile) as f:
    lines = f.read().split("\n")

  lines = list(filter(lambda x: x.strip(), lines))
  lines = np.array(lines)

  img_data = []
  labels = np.array([])

  for x in lines:
    x = x.split("im", 1)[1]
    data, label = x.split("\t", 1)
    label = label.split('\t',1)[0]

    data = np.array(list(map(int, data)))
    img_data.append(data)
    labels = np.append(labels, label)

  return img_data, labels



print("--- OCR ---\n")
print("Importing OCR data, this will take a little while...\n")
ocr_data, ocr_labels = read_ocr('./OCR-data/ocr_train.txt')
ocr_test_data,  ocr_test_labels = read_ocr('./OCR-data/ocr_test.txt')
print("Successfully imported all OCR data\n")

# Standard Perceptron
print("Running Standard Perceptron...\n")
w = np.zeros(128) # initialize weight
mistakes_training = []
mistakes_testing = []

for i in range(20):
  mistakes = 0

  for j, x in enumerate(ocr_data):
    y = ocr_labels[j]
    if y == 'a' or y == 'e' or y == 'i' or y == 'o' or y == 'u':
      y = 1
    else:
      y = -1
    prediction = np.sign(np.dot(w, x))
    if y * prediction <= 0: # mistake was made
      w = w + y * x
      mistakes += 1

  mistakes_training.append(mistakes)

  # Test the weight on testing data
  mistakes = 0

  for j, x in enumerate(ocr_test_data):
    y = ocr_test_labels[j]
    if y == 'a' or y == 'e' or y == 'i' or y == 'o' or y == 'u':
      y = 1
    else:
      y = -1
    prediction = np.sign(np.dot(w, x))
    if y * prediction <= 0: # mistake was made
      mistakes += 1

  mistakes_testing.append(mistakes)


# Avg Perceptron
print("Running Avg Perceptron...\n")
w = np.zeros(128) # initialize weight
w_sum = np.zeros(128)
c = 1 # count

for i in range(20):

  for j, x in enumerate(ocr_data):
    y = ocr_labels[j]
    if y == 'a' or y == 'e' or y == 'i' or y == 'o' or y == 'u':
      y = 1
    else:
      y = -1
    prediction = np.sign(np.dot(w, x))
    if y * prediction <= 0: # mistake was made
      w = w + y * x
      w_sum = w_sum + c * w
      mistakes += 1
      c += 1

avg_w = w_sum / c

# Calculate the training and testing accuracy
mistakes = 0

for j, x in enumerate(ocr_data):
  y = ocr_labels[j]
  if y == 'a' or y == 'e' or y == 'i' or y == 'o' or y == 'u':
    y = 1
  else:
    y = -1
  prediction = np.sign(np.dot(w_sum, x))
  if y * prediction <= 0: # mistake was made
    mistakes += 1

avg_mistakes_training = mistakes

mistakes = 0

for j, x in enumerate(ocr_test_data):
  y = ocr_test_labels[j]
  if y == 'a' or y == 'e' or y == 'i' or y == 'o' or y == 'u':
    y = 1
  else:
    y = -1
  prediction = np.sign(np.dot(w_sum, x))
  if y * prediction <= 0: # mistake was made
    mistakes += 1

avg_mistakes_testing = mistakes



# Output our results
output.write("\nOCR\n\n")
output.write("\t\tMistakes\n")
for i, x in enumerate(mistakes_training):
  output.write(f"Iteration-{i+1}\t{x}\n")

output.write("\n")

num_training_examples = len(ocr_data)
num_testing_examples = len(ocr_test_data)
output.write("\t\tTraining Accuracy\tTesting Accuracy\n")
for i, x in enumerate(mistakes_training):
  output.write(f"Iteration-{i+1}\t{(num_training_examples - x)/num_training_examples}\t{(num_testing_examples - mistakes_testing[i])/num_testing_examples}\n")

output.write("\n")
output.write("Standard Perceptron Training Accuracy\tAvg Perceptron Training Accuracy\n")
output.write(f"{(num_training_examples - mistakes_training[19])/num_training_examples}\t\t\t{(num_training_examples - avg_mistakes_training)/num_training_examples}\n")
output.write("Standard Perceptron Testing Accuracy\tAvg Perceptron Testing Accuracy\n")
output.write(f"{(num_testing_examples - mistakes_testing[19])/num_testing_examples}\t\t\t{(num_testing_examples - avg_mistakes_testing)/num_testing_examples}\n")

output.close()
print("Outputted results to output.txt")