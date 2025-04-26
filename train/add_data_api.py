import json 
from flask import Flask, render_template, redirect
from dotenv import load_dotenv
import os
   
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '../../data/dzs.json')

def adddzs(tag,pt):
    with (open(data_path,'r') )as f:
        dzss=json.load(f)
    t=True
    for i in dzss['intents']:
        if tag==i['tag']:
            t=False
            i['patterns'].append(pt)
            
    if t:        
        dzss['intents'].append({"tag":tag,"patterns":[pt]})
        
    with (open(data_path,'w')) as file:
        json.dump(dzss,file,indent=6)

load_dotenv()

app = Flask(__name__)

app.secret_key = 'ImmortalPotato'

global ktag
ktag='lmao'

@app.route('/')
def inedx():
    global ktag
    return render_template("yuhuh.html",tag=ktag)
    
@app.route('/api/<tag>/dzs/<pat>')
def adddssdzs(tag,pat):
    if pat != 'Pattern':
        adddzs(tag,pat)
    global ktag
    ktag=tag
    return redirect("/")


if __name__ == '__main__':
    app.run()#host='192.168.1.29', port=5000, debug=False)
 