import json 
from flask import Flask, render_template, redirect
from dotenv import load_dotenv

          
def adddzs(tag,pt):
    with (open("C:/Users/pc/Desktop/code/diagnose_ai/data/dzs.json",'r') )as f:
        dzss=json.load(f)
    t=True
    for i in dzss['intents']:
        if tag==i['tag']:
            t=False
            i['patterns'].append(pt)
            
    if t:        
        dzss['intents'].append({"tag":tag,"patterns":[pt]})
        
    with (open("C:/Users/pc/Desktop/code/diagnose_ai/data/dzs.json",'w')) as file:
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
    app.run(debug=False)
     
'''
'''