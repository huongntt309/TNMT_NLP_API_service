import re
from flask import request, Response, Flask, render_template
import torch
from underthesea import sent_tokenize
from waitress import serve
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import itertools


model_classification = None
model_classification2 = None
model_summarization = None
tokenizer_classification = None
tokenizer_summarization = None


# set up functions
def setup():
    # Load 2 model_predictions + 1 model summarization
     
    global model_classification
    global model_classification2
    global model_summarization

    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_cls_path = os.path.join(script_dir, "classification/vit5-cp-6790")
    model_cls_path2 = os.path.join(script_dir, "classification/vit5-cp-3280")
    model_summarization_path = os.path.join(script_dir, "summarization/bartpho-cp11000")

    model_classification = AutoModelForSeq2SeqLM.from_pretrained(model_cls_path)
    model_classification2 = AutoModelForSeq2SeqLM.from_pretrained(model_cls_path2)
    model_summarization = AutoModelForSeq2SeqLM.from_pretrained(model_summarization_path)

    # Load tokenizer
    global tokenizer_classification
    global tokenizer_summarization
    tokenizer_summarization_path = os.path.join(script_dir, "summarization/bartpho-tokenizer")
    tokenizer_vit5_path = os.path.join(script_dir, "classification/vit5-base-tokenizer")
    tokenizer_classification = AutoTokenizer.from_pretrained(tokenizer_vit5_path)
    tokenizer_summarization = AutoTokenizer.from_pretrained(tokenizer_summarization_path)

    if model_classification is None \
            or tokenizer_classification is None:
        return Response(json.dumps({"error": "Model or tokenizer model_classification initialized"}), mimetype='application/json')
    if model_summarization is None \
            or tokenizer_summarization is None:
        return Response(json.dumps({"error": "Model or tokenizer model_summarization initialized"}), mimetype='application/json')
    
    print("Set up model and tokenizer successfully")
    print("Open local link: http://localhost:8080/classification")


class Summarization:
    dict_map = {
        "òa": "oà", "Òa": "Oà", "ÒA": "OÀ", "óa": "oá", "Óa": "Oá", "ÓA": "OÁ", "ỏa": "oả", "Ỏa": "Oả", "ỎA": "OẢ", "õa": "oã", "Õa": "Oã", "ÕA": "OÃ", "ọa": "oạ", "Ọa": "Oạ", "ỌA": "OẠ", "òe": "oè", "Òe": "Oè", "ÒE": "OÈ", "óe": "oé", "Óe": "Oé", "ÓE": "OÉ", "ỏe": "oẻ", "Ỏe": "Oẻ", "ỎE": "OẺ", "õe": "oẽ", "Õe": "Oẽ", "ÕE": "OẼ", "ọe": "oẹ", "Ọe": "Oẹ", "ỌE": "OẸ", "ùy": "uỳ", "Ùy": "Uỳ", "ÙY": "UỲ", "úy": "uý", "Úy": "Uý", "ÚY": "UÝ", "ủy": "uỷ", "Ủy": "Uỷ", "ỦY": "UỶ", "ũy": "uỹ", "Ũy": "Uỹ", "ŨY": "UỸ", "ụy": "uỵ", "Ụy": "Uỵ", "ỤY": "UỴ", "…": "."
    }

    @staticmethod
    def replace_all(text):
        for i, j in Summarization.dict_map.items():
            text = text.replace(i, j)
        return text

    @staticmethod
    def generateSummary(sentnum, title, text):
        text = str(sentnum) + ' câu. Tên: <' + title + '>. Nội dung: <' + text + '>'
        model_summarization.eval()
        with torch.no_grad():
            inputs = tokenizer_summarization(text, max_length=1024, truncation=True, return_tensors='pt')
            outputs = model_summarization.generate(**inputs, max_length=512, num_beams=5,
                                                   early_stopping=True, no_repeat_ngram_size=3)
            prediction = tokenizer_summarization.decode(outputs[0], skip_special_tokens=True)
        return prediction

    @staticmethod
    def getDocSummary(sentnum, title, text, lim=850):
        text, title = Summarization.replace_all(text), Summarization.replace_all(title)
        title_len = len(title.split(' '))
        sentid, prediction = 0, ''
        sents = sent_tokenize(text)
        sentlen = [len(s.split(' ')) for s in sents]

        while sentid < len(sents):
            curlen, curtext = title_len + len(prediction.split(' ')), ''
            while sentid < len(sents) and curlen + sentlen[sentid] <= lim:
                curtext += sents[sentid] + ' '
                curlen += sentlen[sentid]
                sentid += 1

            if sentid < len(sents) and curtext == '':
                curtext += sents[sentid] + ' '
                curlen += sentlen[sentid]
                sentid += 1
            prediction = Summarization.generateSummary(sentnum, title, prediction + ' ' + curtext)
        return prediction



def predict_cls(text):
    # Perform detection
    max_target_length = 256
    inputs = tokenizer_classification(text, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
 
    # model predict
    output_cls = model_classification.generate(
        input_ids=input_ids,
        max_length=max_target_length,
        attention_mask=attention_mask,
    )

    predicted_cls = tokenizer_classification.decode(output_cls[0], skip_special_tokens=True)
    return predicted_cls

def predict_cls2(text):
    # Perform detection
    max_target_length = 256
    inputs = tokenizer_classification(text, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
 
    # model predict
    output_cls = model_classification2.generate(
        input_ids=input_ids,
        max_length=max_target_length,
        attention_mask=attention_mask,
    )

    predicted_cls = tokenizer_classification.decode(output_cls[0], skip_special_tokens=True)
    return predicted_cls

def classify_article(data):
    text = data['title'] + '. ' + data['summary'] + '. ' + data['content']
    is_in_vietnam, province_list = check_in_VietNam(text)
    
    prd_data = predict_cls(text)
    prd_data2 = predict_cls2(text)
    prd_aspect_law = check_aspect_sua_doi_luat(text)
    prd_topic, prd_sentiment, prd_sub_topic, prd_aspect = prd_data.split(';')
    prd_sentiment, prd_sub_topic, prd_aspect = prd_data2.split(';')
        
    print("prd_data:", prd_data)
    print("prd_data2:", prd_data2)
    
    if prd_aspect_law != False:
        prd_aspect = prd_aspect + '. ' + prd_aspect_law
        
    if is_in_vietnam:        
        result = {
            "id"        : data['id'],                          
            "summary"   : data["summary"],          
            "topic"     : prd_topic,                           
            "sub_topic" : prd_sub_topic,                
            "aspect"    : prd_aspect,            
            "sentiment" : prd_sentiment,                      
            "province"  : province_list,
        }
    else:
        result = {
            "id"        : data['id'],                          
            "summary"   : data["summary"],     
            "topic"     : prd_topic,                           
            "sub_topic" : prd_sub_topic,                
            "aspect"    : prd_aspect,            
            "sentiment" : prd_sentiment,                     
            "province"  : "Không",
        }
    return result


def check_in_VietNam(text):
    province_viet_nam_file = "province_viet_nam.txt"

    with open(province_viet_nam_file, 'r', encoding='utf-8') as file:
        provinces = [line.replace("\n", "") for line in file.readlines()]
        
    is_in_vietnam = False
    province_list = set()
    for province in provinces:
        if province in text:
            is_in_vietnam = True
            province_list.add(province)
    province_list = list(province_list)
    
    keywords_TNMT = ['Bộ Tài nguyên và Môi trường', 'Bộ TN&MT'] 
    for kw in keywords_TNMT:
        if kw in text:
            is_in_vietnam = True
            
    return is_in_vietnam, province_list



def check_aspect_sua_doi_luat(text):
    sua_doi_luat_verb = ["sửa đổi", "cập nhật", "thay thế", "bổ sung", "cải tiến", "điều chỉnh", "thay đổi", "chỉnh sửa", "cải cách", "đổi mới"]
    sua_doi_luat_noun = ["nghị định", "luật", "quy định"]

    cartesian_product = list(itertools.product(sua_doi_luat_verb, sua_doi_luat_noun))

    final_rules = []
    for item in cartesian_product:
        final_rules.append(" ".join(item))

    text = text.lower()
    for rule in final_rules:
        if rule in text:
            return rule

    for rule in final_rules:
        if rule in text:
            return rule
    
    return False