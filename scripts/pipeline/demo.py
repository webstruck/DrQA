import bottle
from drqa import pipeline
import json
import pandas as pd
app = bottle.Bottle()
query = []
response = ""
DrQA = pipeline.DrQA(
    cuda=False,
    reader_model="/ml/mfe4ml/raghuvan/nlp/code/DrQA/data/reader/single.mdl",
    ranker_config={'options': {'tfidf_path': "/ml/mfe4ml/raghuvan/nlp/code/DrQA/data/datasets/helpbot/mpp/mpp-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"}},
    db_config={'options': {'db_path': "/ml/mfe4ml/raghuvan/nlp/code/DrQA/data/datasets/helpbot/mpp/mpp.db"}}
)

@app.get("/")
def home():
    with open('/ml/mfe4ml/raghuvan/nlp/code/DrQA/scripts/pipeline/demo.html', 'r') as fl:
        html = fl.read()
        return html

@app.post('/answer')
def answer():
    question = bottle.request.json['question']
    print("received question: {}".format(question))

    global  query, response
    predictions = DrQA.process(
        question, candidates=None, top_n=2, n_docs=5, return_context=True
    )
    dfr = pd.DataFrame(predictions)
    print("[info] RESULTS DF: ")
    print(dfr.describe())
    print("[info] RESULTS: ", json.dumps(dfr.to_json(), indent=4))
    return dfr.to_json()

app.run(port=5050, host='0.0.0.0')