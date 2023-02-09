import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
# from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.legacy import SGD

#  
import tensorflow
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
# documents = []
docs_x = []
docs_y = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # add documents in the corpus
        docs_x.append(word_list)
        docs_y.append(intent['tag'])
        # documents.append((word_list), intent['tag'])

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words =[stemmer.stem(w.lower()) for w in words if w not in ignore_letters]

# words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

labels = sorted(set(classes))

print(len(docs_x), "documents")
print(len(classes), "classes", labels )
print(len(words), "unique lemmatized words", words)

# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))


training = []
output = []
output_empty = [0 for _ in range(len(labels)) ]


for x,doc in enumerate(docs_x):
    bag = []
    
    wrds = [stemmer.stem(word.lower()) for word in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
        
    output_row = output_empty[:]
    output_row[classes.index(docs_y[x])] = 1
    
    training.append(bag)
    output.append(output_row)

    
training = np.array(training)
output = np.array(output)

 
with open('data.pickle', 'wb') as f:
    pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)



model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

model.save('model.tflearn')






# for document in documents:
#     # print("document: ", document)
#     # initialize our bag of words
#     bag = []
#     # list of tokenized words for the pattern
#     pattern_words = document[0]
#     # lemmatize each word - create base word, in attempt to represent related words
#     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
#     # create our bag of words array with 1, if word match found in current pattern
#     for word in words:
#         bag.append(1) if word in pattern_words else bag.append(0)

#     output_row = list(output_empty)
#     output_row[classes.index(document[1])] = 1

#     training.append([bag, output_row])



# # shuffle our features and turn into np.array
# random.shuffle(training)

# training = np.array(training)





# # create train and test lists. X - patterns, Y - intents
# train_x = list(training[:,0])
# train_y = list(training[:,1])
# print("Training data created")


# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))

# # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# # fitting and saving the model
# hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# model.save('chatbot_model.h5', hist)

# print("model created")






