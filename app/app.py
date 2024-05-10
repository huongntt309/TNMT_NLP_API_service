from flask import request, Response, Flask, render_template
from waitress import serve
import json
import os
from sp_func import setup, Classification, Summarization


app = Flask(__name__)



@app.route("/")
def root():
    return Response(json.dumps({"Application": "TNMT api service"}), mimetype='application/json')


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
                    "anchor"    :"VOV.VN - ... quốc gia.",
                    "content"   :"Phát biểu ...kinh nghiệm."
                }
            ]
        Return: request.json 
            [
                {
                    "id"        :"abc123",                               
                    "summary"   :"Sáng 4/5, ... quốc gia"               
                    "topic"     :"yes",                                  
                    "sub_topic" :"tài nguyên đất",                      
                    "aspect"    :"chính sách quản lý",                   
                    "sentiment" :"tích cực",                             
                    "province"  : ["Hà Nội", "Hồ Chí Minh", "Bắc Giang"] 
                },
            ]
    """
    # Load the corpus from the request
    array_data = request.json
    array_results = []
    
    for data in array_data:
        cls_data = Classification.classify_article(data)
        array_results.append(cls_data)
    
    return Response(json.dumps(array_results), mimetype='application/json')

@app.route("/summarize", methods=["POST"])
def summarize_api():
    """
        Handler of /summarize POST endpoint
        Input: request.json 
            ex: 
            [
                {
                    "id"        :"abc123",
                    "title"     :"Thủ tướng chủ trì Phiên họp Chính phủ thường kỳ tháng 4",
                    "anchor"    :"VOV.VN - ... quốc gia.",
                    "content"   :"Phát biểu ...kinh nghiệm."
                }
            ]
        Return: request.json 
            [
                {
                    "id"        :"abc123",                               
                    "summary"   :"Sáng 4/5, ... quốc gia"                
                    "topic"     :"yes",                                  
                    "sub_topic" :"tài nguyên đất",                       
                    "aspect"    :"chính sách quản lý",                   
                    "sentiment" :"tích cực",                             
                    "province"  : ["Hà Nội", "Hồ Chí Minh", "Bắc Giang"] 
                },
            ]
    """
    # Load the corpus from the request
    data = request.json
   
    # decode output
    summary = Summarization.getDocSummary(data, sentnum=3)
    
    return Response(json.dumps(summary), mimetype='application/json')


@app.route("/sum_cls", methods=["POST"])
def sum_cls_api():
    """
        Handler of /sum_cls POST endpoint
        Input: request.json 
            ex: 
            [
                {
                    "id"        :"abc123",
                    "title"     :"Thủ tướng chủ trì Phiên họp Chính phủ thường kỳ tháng 4",
                    "anchor"    :"VOV.VN - ... quốc gia.",
                    "content"   :"Phát biểu ...kinh nghiệm."
                }
            ]
        Return: request.json 
            [
                {
                    "id"        :"abc123",                               
                    "summary"   :"Sáng 4/5, ... quốc gia"                
                },
            ]
    """
    # Load the corpus from the request
    array_data = request.json
   
    # Summarization
    array_summary = Summarization.getDocSummary(array_data, sentnum=5)
    
    # Classification
    array_cls = []
    
    for data in array_data:
        cls_data = Classification.classify_article(data)
        array_cls.append(cls_data)
    
    # merge the results
    array_results = []
    for cls_object in array_cls:
        cls_id = cls_object['id']
        result = next((result for result in array_summary if result['id'] == cls_id), None)
        if result:
            array_results.append({
                "id"        : cls_id,
                "summary"   : result['summary'],
                "topic"     : cls_object["topic"],                           
                "sub_topic" : cls_object["sub_topic"],                
                "aspect"    : cls_object["aspect"],            
                "sentiment" : cls_object["sentiment"],                      
                "province"  : cls_object["province"],
            })
    # return json.dumps(array_results)
    return Response(json.dumps(array_results), mimetype='application/json')


# Call the setup function before starting the server
setup()

serve(app, host='0.0.0.0', port=8081)
