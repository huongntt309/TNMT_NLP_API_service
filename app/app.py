from collections import defaultdict
from flask import request, Response, Flask, render_template
from waitress import serve
import json
import os
from sp_func import setup, Classification, Summarization


app = Flask(__name__)

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    index_html_path = os.path.join(current_dir, "index.html")
    with open(index_html_path, encoding="utf-8") as file:
        html_content = file.read()
    return html_content


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
    array_data = request.json
    
    print("Classifying and Summarizing ...")
    print("Phase 1: Classifying ...")
    
    # Classification
    array_cls = Classification.classify_article(array_data)
    
    print("Phase 2: Summarizing ...")
    # Process label 0 object 
    array_data_summ = []

    for idx, data in enumerate(array_data):
        # Checking if the 'topic' index exists in array_cls
        if idx < len(array_cls):
            # Assuming you want to check if the 'topic' value is Có
            if array_cls[idx]["topic"] == "Có" and array_cls[idx]["id"] == data["id"]:
                array_data_summ.append(data)

    # Summarization
    object_summary = Summarization.getDocSummary(array_data_summ, sentnum=3)
    # merge the Summarization and Classification results
    array_results = []
    for cls_object in array_cls:
        cls_id = cls_object['id']
        array_results.append({
            "id"        : cls_id,
            "summary"   : object_summary[cls_id] ,
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
