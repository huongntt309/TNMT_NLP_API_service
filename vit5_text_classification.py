import re
from flask import request, Response, Flask, render_template
import torch
from underthesea import sent_tokenize
from waitress import serve
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from sp_func import check_in_VietNam, setup, classify_article, Summarization


app = Flask(__name__)




@app.route("/classification")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    index_html_path = os.path.join(current_dir, "index_cls.html")
    with open(index_html_path, encoding="utf-8") as file:
        html_content = file.read()
    return html_content

@app.route("/summarization")
def summarization_page():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    index_html_path = os.path.join(current_dir, "index_sum.html")
    with open(index_html_path, encoding="utf-8") as file:
        html_content = file.read()
    return html_content

@app.route("/predict", methods=["POST"])
def predict_api():
    """
        Handler of /detect POST endpoint
        Input: request.json 
            ex: 
            [
                {
                    "id"        :"abc123",
                    "title"     :"Thủ tướng chủ trì Phiên họp Chính phủ thường kỳ tháng 4",
                    "summary"   :"VOV.VN - ... quốc gia.",
                    "content"   :"Phát biểu ...kinh nghiệm."
                }
            ]
        Return: request.json 
            [
                {
                    "id"        :"abc123",                               - lấy ở input
                    "summary"   :"Sáng 4/5, ... quốc gia"                - giữ nguyên
                    "topic"     :"yes",                                  - đã làm được - model 4
                    "sub_topic" :"tài nguyên đất",                       - thêm "luật" - model 4
                    "aspect"    :"chính sách quản lý",                   - đã làm được - model 4
                    "sentiment" :"tích cực",                             - đã làm được - model 4
                    "province"  : ["Hà Nội", "Hồ Chí Minh", "Bắc Giang"] - đã làm được - rule 
                },
            ]
    """
    # Load the corpus from the request
    array_data = request.json
    array_results = []
    
    for data in array_data:
        cls_data = classify_article(data)
        array_results.append(cls_data)
    
    print(array_results)
    # return json.dumps(array_results)
    return Response(json.dumps(array_results), mimetype='application/json')

@app.route("/summarize", methods=["POST"])
def summarize_api():
    """
        Handler of /detect POST endpoint
        Input: corpus
        Return: type of corpus
    """
    # Load the corpus from the request
    data = request.json
    corpus = data.get('corpus', '')  # Extracting the 'corpus' field from the JSON data
    title = data.get('title', '')  # Extracting the 'title' field from the JSON data
    numSent = data.get('numSent', '')  
   
    # decode output
    summary = Summarization.getDocSummary(numSent, title, corpus)
   
    print("model_summarization:", summary)

    result = {
        "summary": summary,
    }

    return Response(json.dumps(result), mimetype='application/json')





# Call the setup function before starting the server
setup()

serve(app, host='0.0.0.0', port=8080)
