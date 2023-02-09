import random
import json
from flask_cors import CORS
import torch
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Hank"

app = Flask(__name__)

CORS(app)

application = app

scored = 0
not_answered = 0

@app.route("/")
def hello():
    return "Congratulation! You service Hosted on CPanel"

@app.route('/hankapi', methods = ['GET', 'POST'])
def get_response():
    global scored
    global not_answered

    msg = request.args.get('message')
    print(msg)
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                scored += 1
                
                print("scored   ",scored)
                return jsonify(str(random.choice(intent['responses'])),tag) 
    not_answered += 1
    return jsonify("Sorry, I don't understand your question.", 0)

  
# this is the API to get the score based on the questions asked by user
@app.route('/score',methods = ['GET', 'POST'])
def score():
    print("scored   ",scored)
    return {"not_answered":not_answered, "top_3":scored}
        
@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

if __name__ == "__main__":
    app.run("0.0.0.0")

