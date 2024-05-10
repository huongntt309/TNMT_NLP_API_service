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
model_classification_subtopic = None
model_summarization = None
tokenizer_classification = None
tokenizer_summarization = None


# set up functions
def setup():
    # Load 2 model_predictions + 1 model summarization
     
    global model_classification
    global model_classification_subtopic
    global model_summarization

    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_cls_path = os.path.join(script_dir, "classification/vit5-cp-6660")
    model2_cls_path = os.path.join(script_dir, "classification/subtopic-5710")
    model_summarization_path = os.path.join(script_dir, "summarization/bartpho-cp11000")

    model_classification = AutoModelForSeq2SeqLM.from_pretrained(model_cls_path)
    model_classification_subtopic = AutoModelForSeq2SeqLM.from_pretrained(model2_cls_path)
    model_summarization = AutoModelForSeq2SeqLM.from_pretrained(model_summarization_path)

    # Load tokenizer
    global tokenizer_classification
    global tokenizer_summarization
    tokenizer_summarization_path = os.path.join(script_dir, "summarization/bartpho-tokenizer")
    tokenizer_vit5_path = os.path.join(script_dir, "classification/vit5-base-tokenizer")
    tokenizer_classification = AutoTokenizer.from_pretrained(tokenizer_vit5_path)
    tokenizer_summarization = AutoTokenizer.from_pretrained(tokenizer_summarization_path)

    if model_classification is None \
        or model_classification_subtopic is None \
            or tokenizer_classification is None:
        return Response(json.dumps({"error": "Model or tokenizer model_classification initialized"}), mimetype='application/json')
    if model_summarization is None \
            or tokenizer_summarization is None:
        return Response(json.dumps({"error": "Model or tokenizer model_summarization initialized"}), mimetype='application/json')
    
    print("Set up model and tokenizer successfully")
    print("Open local link: http://localhost:8080/classification")


class Summarization:
    dict_map_path_json = 'bow_folder/dict_map.json'
    with open(dict_map_path_json, 'r', encoding='utf-8') as f:
        dict_map = json.load(f)

    @staticmethod
    def replace_all(text):
        for i, j in Summarization.dict_map.items():
            text = text.replace(i, j)
        return text

    @staticmethod
    def generateSummary(sentnum, texts):
        model_summarization.eval()
        with torch.no_grad():
            inputs = tokenizer_summarization(texts, padding=True, max_length=1024, truncation=True, return_tensors='pt')
            outputs = model_summarization.generate(**inputs, max_length=2048, num_beams=5,
                                    early_stopping=True, no_repeat_ngram_size=3)
            prediction = tokenizer_summarization.batch_decode(outputs, skip_special_tokens=True)
        return prediction

    @staticmethod
    def divideText(title_len, prediction, sents, lim=850):
        sentlen = [len(s.split(' ')) for s in sents]
        sentid = 0
        curlen, curtext = title_len + len(prediction.split(' ')), ''
        
        while sentid < len(sents) and curlen + sentlen[sentid] <= lim:
            curtext += sents[sentid] + ' '
            curlen += sentlen[sentid]
            sentid += 1

        if sentid < len(sents) and curtext == '':
            curtext += sents[sentid] + ' '
            curlen += sentlen[sentid]
            sentid += 1
        return curtext, sents[sentid:]

    @staticmethod
    def getDocSummary(docs, sentnum):
        '''
        INPUT:
            docs: json data
            [
                {
                    id:
                    title:
                    anchor:
                    content:
                },
                {}
            ]
        OUTPUT:
            res:
            [
                {
                    id:
                    summary:
                },
                {}
            ] 
        '''
        sents, titles, title_lens, res = {},{},{},[]
        batch_size = 8
        
        for d in docs:
            sents[d['id']] = sent_tokenize(Summarization.replace_all(d['anchor'] + '.\n' + d['content']))
            titles[d['id']] = Summarization.replace_all(d['title'])
            title_lens[d['id']] = len(titles[d['id']].split(' '))
        
        for i in range(0, len(docs), batch_size):
            batchIDs = list(titles.keys())[i:i+batch_size]
            prediction_b = {i:'' for i in batchIDs}
            while len(batchIDs):
                text_b = []
                for ID in batchIDs:
                    text, sents[ID] = Summarization.divideText(title_lens[ID], prediction_b[ID], sents[ID])
                    text_b.append(str(sentnum) + ' câu. Tên: <' + titles[ID] + '>. Nội dung: <' + prediction_b[ID] + ' ' + text + '>')

                summs = Summarization.generateSummary(sentnum, text_b)
                removeIDs = []
                for i, ID in enumerate(batchIDs):
                    prediction_b[ID] = summs[i]
                    if sents[ID] == []:
                        res.append({'id': ID, 'summary': summs[i]})
                        removeIDs.append(ID)
                
                batchIDs = [i for i in batchIDs if i not in removeIDs]
        return res


class Classification:
    
    @staticmethod
    def predict_cls(text):
        def preprocess_text(text):
            # remove redundant spaces
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text

        text = preprocess_text(text)
        # Perform detection
        max_target_length = 256
        inputs = tokenizer_classification(text, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
    
        # model predict 4
        output_cls = model_classification.generate(
            input_ids=input_ids,
            max_length=max_target_length,
            attention_mask=attention_mask,
        )
        
        # model predict subtopic
        output_cls_subtopic = model_classification_subtopic.generate(
            input_ids=input_ids,
            max_length=max_target_length,
            attention_mask=attention_mask,
        )
        predicted_cls = tokenizer_classification.decode(output_cls[0], skip_special_tokens=True)
        predicted_subtopic = tokenizer_classification.decode(output_cls_subtopic[0], skip_special_tokens=True)
        return predicted_cls, predicted_subtopic

    @staticmethod
    def classify_article(data):
        text = data['title'] + '. ' + data['anchor'] + '. ' + data['content']
        is_in_vietnam, province_list = Classification.check_in_VietNam(text)
        
        prd_data, prd_subtopic = Classification.predict_cls(text)
        prd_topic, prd_sentiment, _, prd_aspect = prd_data.split(';')
        prd_aspect_law = Classification.check_aspect_law(text)
        
        if prd_topic == "Không":
            prd_subtopic = "Không"
            
        if prd_topic != "Không":
            prd_aspect = [prd_aspect]
            
            if prd_aspect_law != False :
                prd_aspect.append(prd_aspect_law)
            
        result = {
            "id"        : data['id'],                          
            "topic"     : prd_topic,                           
            "sub_topic" : prd_subtopic,                
            "aspect"    : prd_aspect,            
            "sentiment" : prd_sentiment,                      
            "province"  : province_list,
        }
        return result

    @staticmethod
    def check_in_VietNam(text):
        province_viet_nam_file = "bow_folder/province_viet_nam.txt"

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


    @staticmethod
    def check_aspect_law(text):
        # open file
        law_file = "bow_folder/aspect_law.txt"

        with open(law_file, 'r', encoding='utf-8') as file:
            law_names = [line.strip() for line in file.readlines()]

        # count frequency of law keywords
        text = text.lower()
        total_frq_law = 0

        for name in law_names:
            count = text.count(name.lower())
            total_frq_law += count
        
        # calculate length/total frequency
        threshold_law_frq = 10
        threshold_leng_law_frq = 100
        if total_frq_law > threshold_law_frq:
            text_leng = len(text.split(' '))
            ratio = text_leng/total_frq_law
            if ratio < threshold_leng_law_frq:
                return "Luật sửa đổi"
        else:
            return False
        
