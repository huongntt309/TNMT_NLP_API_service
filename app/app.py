from flask import request, Response, Flask
from waitress import serve
import json
from sp_func import setup, Classification, Summarization


app = Flask(__name__)



@app.route("/")
def root():
    return Response(json.dumps({"Application": "TNMT api service"}), mimetype='application/json')


@app.route("/classify", methods=["POST"])
def classify_api():
    """
       Input, Output described in README.md
    """
    print("Classifying ...")
    
    # Load the corpus from the request
    array_data = request.json
    array_results = []
    
    for data in array_data:
        cls_data = Classification.classify_article(data)
        cls_data["summary"] = ""
        array_results.append(cls_data)
    
    print("Classification process has finished")
    
    return Response(json.dumps(array_results), mimetype='application/json')

@app.route("/summarize", methods=["POST"])
def summarize_api():
    """
       Input, Output described in README.md
    """
    print("Summarizing ...")
    
    # Load the corpus from the request
    data = request.json
   
    # decode output
    summary_results = Summarization.getDocSummary(data, sentnum=3)
    
    # Add empty fields to each object in summary_results
    for result in summary_results:
        result["topic"] = ""
        result["sub_topic"] = ""
        result["aspect"] =  ""
        result["sentiment"] = ""
        result["province"] = []
    
    print("Summarization process has finished")
    return Response(json.dumps(summary_results), mimetype='application/json')


@app.route("/sum-cls", methods=["POST"])
def sum_cls_api():
    """
       Input, Output described in README.md
    """
    # Load the corpus from the request
    array_data, array_data_summ = request.json, []
    
    print("Classifying and Summarizing ...")
    
    # Classification
    array_cls = []
    
    for data in array_data:
        cls_data = Classification.classify_article(data)
        array_cls.append(cls_data)
        if cls_data['topic'] is not 'Không':
            array_data_summ.append(data)

    # Summarization
    array_summary = Summarization.getDocSummary(array_data_summ, sentnum=5)
    
    # merge the results
    array_results = []
    for cls_object in array_cls:
        cls_id = cls_object['id']
        array_results.append({
            "id"        : cls_id,
            "summary"   : array_summary[cls_id] if cls_object["topic"] is not 'Không' else None,
            "topic"     : cls_object["topic"],
            "sub_topic" : cls_object["sub_topic"],
            "aspect"    : cls_object["aspect"],
            "sentiment" : cls_object["sentiment"],
            "province"  : cls_object["province"],
        })
            
    print("Classification and Summarization process has finished")
    return Response(json.dumps(array_results), mimetype='application/json')


# Call the setup function before starting the server
setup()

serve(app, host='0.0.0.0', port=5000)
