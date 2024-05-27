import ast
from collections import defaultdict
import re
from flask import Response
import torch
from underthesea import sent_tokenize
from waitress import serve
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import platform

model_classification = None
model_classification_subtopic = None
model_summarization = None
tokenizer_classification = None
tokenizer_summarization = None
device = None

def setup_device():
    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cuda_version = torch.version.cuda
        print(f"Setup device: CUDA (version {cuda_version})")
    else:
        device = torch.device('cpu')
        print("Setup device: CPU")

# set up functions
def setup():
    # Load 2 model_predictions + 1 model summarization
    # transformer_cache()
    setup_device()
    
    global model_classification
    global model_classification_subtopic
    global model_summarization

    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_cls_path = os.path.join(script_dir, "classification/vit5-cp-6660")
    model2_cls_path = os.path.join(script_dir, "classification/subtopic-5710")
    model_summarization_path = os.path.join(script_dir, "summarization/bartpho-cp11000")

    model_classification = AutoModelForSeq2SeqLM.from_pretrained(model_cls_path).to(device)
    model_classification_subtopic = AutoModelForSeq2SeqLM.from_pretrained(model2_cls_path).to(device)
    model_summarization = AutoModelForSeq2SeqLM.from_pretrained(model_summarization_path).to(device)

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


class Summarization:
    dict_map_path_json = 'bow_folder/dict_map.json'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    dict_map_path = os.path.join(script_dir, dict_map_path_json)
    with open(dict_map_path, 'r', encoding='utf-8') as f:
        dict_map = json.load(f)

    @staticmethod
    def replace_all(text):
        for i, j in Summarization.dict_map.items():
            text = text.replace(i, j)
        return text

    @staticmethod
    def generateSummary(texts):
        model_summarization.eval()
        with torch.no_grad():
            inputs = tokenizer_summarization(texts, padding=True, max_length=1024, truncation=True, return_tensors='pt').to(device)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model_summarization.generate(**inputs, max_length=1024, num_beams=5,
                                    early_stopping=True, no_repeat_ngram_size=3)
            prediction = tokenizer_summarization.batch_decode(outputs, skip_special_tokens=True)
        return prediction

    @staticmethod
    def divideText(title_len, prediction, sentlen, lim=900):
        sentid = 0
        curlen = title_len + len(prediction.split(' '))
        while sentid < len(sentlen) and curlen + sentlen[sentid] <= lim:
            curlen += sentlen[sentid]
            sentid += 1

        if sentid < len(sentlen) and sentid == 0:
            curlen += sentlen[sentid]
            sentid += 1
        return sentid

    @staticmethod
    def getDocSummary(docs, sentnum):
        '''
        INPUT:
            [
                {
                    id:
                    title:
                    anchor:
                    content:
                }
            ]
        OUTPUT:
            {
                id: summary
            }
        '''

        sents, titles, title_lens, sent_lens = {}, {}, {}, {}
        res = defaultdict(str)
        batch_size = 4

        for d in docs:
            sents[d['id']] = sent_tokenize(Summarization.replace_all(d['anchor'] + '.\n' + d['content']))
            titles[d['id']] = Summarization.replace_all(d['title'])
            title_lens[d['id']] = len(titles[d['id']].split(' '))
            sent_lens[d['id']] = [len(s.split(' ')) for s in sents[d['id']]]

        docs = sorted(docs, key=lambda x: sum(sent_lens[x['id']]) + title_lens[d['id']])

        for i in range(0, len(docs), batch_size):
            batchIDs = [d['id'] for d in docs[i:i + batch_size]]
            prediction_b = {i: '' for i in batchIDs}
            while len(batchIDs):
                text_b = []
                for ID in batchIDs:
                    nextID = Summarization.divideText(title_lens[ID], prediction_b[ID], sent_lens[ID])
                    text = ' '.join(sents[ID][:nextID])
                    sents[ID], sent_lens[ID] = sents[ID][nextID:], sent_lens[ID][nextID:]
                    text_b.append(str(sentnum) + ' câu. Tên: <' + titles[ID] + '>. Nội dung: <' + prediction_b[ID] + ' ' + text + '>')

                summs = Summarization.generateSummary(text_b)
                removeIDs = []
                for ii, ID in enumerate(batchIDs):
                    prediction_b[ID] = summs[ii]
                    if sents[ID] == []:
                        res[ID] = summs[ii]
                        removeIDs.append(ID)

                batchIDs = [i for i in batchIDs if i not in removeIDs]
        return res


class Classification:
    @staticmethod
    def predict_cls(texts):
        def preprocess_text(text):
            # remove redundant spaces
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text

        processed_text = [preprocess_text(text) for text in texts]
        # Perform detection
        max_target_length = 256
        inputs = tokenizer_classification(processed_text, max_length=1024, truncation=True, padding=True ,return_tensors="pt")
        
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # model predict 4
        output_cls = model_classification.generate(
            input_ids=input_ids,
            max_length=max_target_length,
            attention_mask=attention_mask,
        )
        predicted_cls = [tokenizer_classification.decode(out, skip_special_tokens=True) for out in output_cls]
        
        tnmt_indices = [index for index, value in enumerate(predicted_cls) if value != "Không;Không;Không;Không"]

        if len(tnmt_indices) > 0:
            tnmt_indices = torch.tensor(tnmt_indices).to(device)
            selected_input_ids_tensor = torch.index_select(input_ids, 0, tnmt_indices)
            selected_attention_mask_tensor = torch.index_select(attention_mask, 0, tnmt_indices)

            # model predict subtopic
            output_cls_subtopic = model_classification_subtopic.generate(
                input_ids=selected_input_ids_tensor.to(device),
                max_length=max_target_length,
                attention_mask=selected_attention_mask_tensor.to(device),
            )

            predicted_subtopic = [tokenizer_classification.decode(out, skip_special_tokens=True) for out in output_cls_subtopic]

            predicted_subtopic_final = ["Không"] * len(texts)

            for i, idx in enumerate(tnmt_indices):
                predicted_subtopic_final[idx] = predicted_subtopic[i]

            return predicted_cls, predicted_subtopic_final
        else:
            predicted_subtopic_final = ["Không"] * len(texts)
            return predicted_cls, predicted_subtopic_final

    @staticmethod
    def classify_article(data):
        '''
        Input:
            data: a list of objects containing id, title, anchor, content.
        Ouptut:
            results: a list of objects containing id, title, summary, and other cls information.
        '''
        results = []
        batch_size = 4
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        # for each batch
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data))
            batch = data[start_idx:end_idx]
            batch_data = []
            for text in batch:
                batch_data.append(text['title'] + '. ' + text['anchor'] + '. ' + text['content'])
            
            prd_data_batch, prd_subtopic_batch = Classification.predict_cls(batch_data)
            
            # For each object individual
            for j in range(len(batch_data)):
                # Get the object individual
                object_data = batch[j]
                text_data = batch_data[j]
                prd_data_model4, prd_subtopic_model1 = prd_data_batch[j], prd_subtopic_batch[j]
                # Process for each object
                try:
                    prd_topic, prd_sentiment, prd_subtopic_model4, prd_aspect = prd_data_model4.split(';')
                except:
                    result = {
                        "id"        : object_data['id'],                          
                        "topic"     : "Không - Exceptions",                           
                        "sub_topic" : [],                
                        "aspect"    : [],         
                        "sentiment" : "Không",                      
                        "province"  : [],
                    }
                    results.append(result)
                    continue
                 
                if prd_topic == "Không":
                    result = {
                        "id"        : object_data['id'],                          
                        "topic"     : "Không",                           
                        "sub_topic" : [],                
                        "aspect"    : [],         
                        "sentiment" : "Không",                      
                        "province"  : [],
                    }
                    results.append(result)
                    
                elif prd_topic != "Không":
                    # Provinces
                    is_in_vietnam, province_list = Classification.check_VietNam_provinces(text_data)
                    
                    # Aspect Law Change
                    prd_aspect_law = Classification.check_aspect_law(text_data)
                    prd_aspect = [prd_aspect]
                    if prd_aspect_law != False :
                        prd_aspect.append(prd_aspect_law)
                    
                    # Join 2 model subtopic predictions
                    prd_subtopic = ast.literal_eval(prd_subtopic_model1)

                    # Rename the same subtopic (of 2 model viT5)
                    if prd_subtopic_model4 == "chung":
                        prd_subtopic_model4 = "Thông tin chung"

                    if prd_subtopic_model4.lower() not in map(str.lower, prd_subtopic):
                        prd_subtopic.append(prd_subtopic_model4)
      
                    result = {
                        "id": object_data['id'],
                        "topic": prd_topic,
                        "sub_topic": prd_subtopic,
                        "aspect": prd_aspect,
                        "sentiment": prd_sentiment,
                        "province": province_list,
                    }
                    results.append(result)
                    
        return results

    @staticmethod
    def check_VietNam_provinces(text):
        province_viet_nam_file = "bow_folder/province_viet_nam.txt"
        script_dir = os.path.dirname(os.path.realpath(__file__))
        province_viet_nam_path = os.path.join(script_dir, province_viet_nam_file)
        
        with open(province_viet_nam_path, 'r', encoding='utf-8') as file:
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
        
        script_dir = os.path.dirname(os.path.realpath(__file__))
        law_file_path = os.path.join(script_dir, law_file)

        with open(law_file_path, 'r', encoding='utf-8') as file:
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
        
