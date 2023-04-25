
import os
import json
import re


def judge_true(data):
    ground_truth=data['Instance']["ground_truth"].split("\n")
    ground_truth_dict={}
    for term in ground_truth:
        if "polarity" in term:
            if "positive" in term:
                polarity="positive"
            elif "negative" in term:
                polarity="negative"
            else:
                polarity="neutral"
            aspect_index=re.findall("aspect\d",term)[0]  #aspect1
            ground_truth_dict[aspect_index]=polarity
            
    predict=data["Prediction"].split()
    predict_dict={}
    for predict_term in predict:
        if "polarity" in predict_term:
            if "positive" in predict_term:
                predict_polarity="positive"
            elif "negative" in predict_term:
                predict_polarity="negative"
            else:
                predict_polarity="neutral"
            predict_index=re.findall("aspect\d",predict_term)[0]
            predict_dict[predict_index]=predict_polarity
    for key in ground_truth_dict:
        if key in predict_dict.keys() and ground_truth_dict[key]==predict_dict[key]:
            return True
        else:
            return False


def caculate_all_acc(output_path):

    predict_path=os.path.join(output_path, "predict_eval_predictions.jsonl")
    with open(predict_path, 'r', encoding='utf-8') as f:
        total_aspect=0
        total_truth=0
        yiluo=0
        neutral=0
        
        for line in f:
            data=json.loads(line)
            ground_truth=data['Instance']["ground_truth"].split("\n")
            ground_truth_dict={}
            for term in ground_truth:
                if "polarity" in term:
                    if "positive" in term:
                        polarity="positive"
                    elif "negative" in term:
                        polarity="negative"
                    else:
                        polarity="neutral"
                    aspect_index=re.findall("aspect\d",term)[0]  #aspect1
                    ground_truth_dict[aspect_index]=polarity
                
                    
            predict=data["Prediction"].split()
            predict_dict={}
            for predict_term in predict:
                if "polarity" in predict_term:
                    if "positive" in predict_term:
                        predict_polarity="positive"
                    elif "negative" in predict_term:
                        predict_polarity="negative"
                    else:
                        predict_polarity="neutral"
                    predict_index=re.findall("aspect\d",predict_term)[0]
                    predict_dict[predict_index]=predict_polarity
            for key in ground_truth_dict:
                total_aspect+=1
                if predict_dict[key]=='neutral':
                    neutral+=1
                if key in predict_dict.keys() and ground_truth_dict[key]==predict_dict[key]:
                    total_truth+=1
                elif key not in predict_dict.keys():
                    yiluo+=1
                # else:
                    # print(line)
    print(neutral)
    print(total_aspect)
    print(total_truth/total_aspect)

def caculate_acc(test_dict,name):
    total_aspect=0
    total_truth=0
    yiluo=0
    
    for key in test_dict.keys():
        data=test_dict[key]
        ground_truth=data['Instance']["ground_truth"].split("\n")
        ground_truth_dict={}
        for term in ground_truth:
            if "polarity" in term:
                polarity="positive" if "positive" in term else "negative"
                aspect_index=re.findall("aspect\d",term)[0]  #aspect1
                ground_truth_dict[aspect_index]=polarity
            
                
        predict=data["Prediction"].split()
        predict_dict={}
        for predict_term in predict:
            if "polarity" in predict_term:
                predict_polarity="positive" if "positive" in predict_term else "negative"
                predict_index=re.findall("aspect\d",predict_term)[0]
                predict_dict[predict_index]=predict_polarity
        for key in ground_truth_dict:
            total_aspect+=1
            if key in predict_dict.keys() and ground_truth_dict[key]==predict_dict[key]:
                total_truth+=1
            elif key not in predict_dict.keys():
                yiluo+=1

    print("Total aspect of {}:{}".format(name,total_aspect))
    print("ACC:{}\n".format(total_truth/total_aspect))

def caculate_overall_acc(output_path):
    #判断三种变形下都对的比例
    predict_path=os.path.join(output_path, "predict_eval_predictions.jsonl")
    with open(predict_path, 'r', encoding='utf-8') as f:
        Origin={}
        RevTgt={}
        RevNon={}
        AddDiff={}        
        for line in f:
            data=json.loads(line)
            id=data['Instance']['id']
            if 'adv1' in id:
                RevTgt[id]=data
            elif 'adv2' in id:
                RevNon[id]=data
            elif 'adv4' in id:
                AddDiff[id]=data
            else:
                Origin[id]=data
        total_sample=len(Origin)
        total_truth=0
        for key in Origin.keys():
            flag=True
            samples=[]
            samples.append(Origin[key])
            for revtgt_key in RevTgt.keys():
                if key in revtgt_key:
                    samples.append(RevTgt[revtgt_key])
            for revnon_key in RevNon.keys():
                if key in revnon_key:
                    samples.append(RevNon[revnon_key])
            for adddiff_key in AddDiff.keys():
                if key in adddiff_key:
                    samples.append(AddDiff[adddiff_key])
            for sample in samples:
                if not judge_true(sample):
                    flag=False
                    # print(sample)
                    break
            if flag==True:
                total_truth+=1
            
    print(total_truth)
    print(total_truth/total_sample)
                    
                

def caculate_acc_sep(output_path):
    predict_path=os.path.join(output_path, "predict_eval_predictions.jsonl")
    with open(predict_path, 'r', encoding='utf-8') as f:
        
        Origin={}
        RevTgt={}
        RevNon={}
        AddDiff={}
        RevTgt_Origin={}
        RevNon_Origin={}
        AddDiff_Origin={}
        
        for line in f:
            data=json.loads(line)
            id=data['Instance']['id']
            if 'adv1' in id:
                RevTgt[id]=data
            elif 'adv2' in id:
                RevNon[id]=data
            elif 'adv4' in id:
                AddDiff[id]=data
            else:
                Origin[id]=data
        for key in RevTgt.keys():
            origin_key=key.split('_adv1')[0]
            RevTgt_Origin[origin_key]=Origin[origin_key]
        for key in RevNon.keys():
            origin_key=key.split('_adv2')[0]
            RevNon_Origin[origin_key]=Origin[origin_key]
        for key in AddDiff.keys():
            origin_key=key.split('_adv4')[0]
            AddDiff_Origin[origin_key]=Origin[origin_key]
    caculate_acc(Origin,"Origin")
    caculate_acc(RevTgt,"RevTgt")
    caculate_acc(RevNon,"RevNon")
    caculate_acc(AddDiff,"AddDiff")
    caculate_acc(RevTgt_Origin, "RevTgt_Origin")
    caculate_acc(RevNon_Origin,"RevNon_origin")
    caculate_acc(AddDiff_Origin,"AddDiff_origin")
         
def caculate_error_num(output_path):
    sentence2id={}
    with open("/workspace/ABSA/Laptop/trans_RevNon_my.json","r") as f:
        s = f.read()
        f = json.loads(s)
        for key in f.keys():
            sample=f[key]
            sentence2id[sample['sentence']]=sample['sample_id']

    predict_path=os.path.join(output_path, "predict_eval_predictions.jsonl")
    with open(predict_path, 'r', encoding='utf-8') as f:
        total_aspect=0
        total_truth=0
        yiluo=0
        error_sample=[]
        
        for line in f:
            data=json.loads(line)
            ground_truth=data['Instance']["ground_truth"].split("\n")
            ground_truth_dict={}
            for term in ground_truth:
                if "polarity" in term:
                    polarity="positive" if "positive" in term else "negative"
                    aspect_index=re.findall("aspect\d",term)[0]  #aspect1
                    ground_truth_dict[aspect_index]=polarity
                
                    
            predict=data["Prediction"].split()
            predict_dict={}
            for predict_term in predict:
                if "polarity" in predict_term:
                    predict_polarity="positive" if "positive" in predict_term else "negative"
                    predict_index=re.findall("aspect\d",predict_term)[0]
                    predict_dict[predict_index]=predict_polarity
            for key in ground_truth_dict:
                total_aspect+=1
                if key in predict_dict.keys() and ground_truth_dict[key]==predict_dict[key]:
                    total_truth+=1
                else:
                    sample_id=sentence2id[data['Instance']['sentence']]
                    if sample_id not in error_sample:
                        error_sample.append(sample_id)
    print(error_sample)
    print(len(error_sample))
    



if __name__ == '__main__':
    output_path="/workspace/output/flan-t5-large-ABSA-laptop-all-annotion"
    caculate_all_acc(output_path)  #变形后的test set的acc
    # caculate_error_num(output_path)
    caculate_acc_sep(output_path)   #计算每种变形下的ACC变化
    caculate_overall_acc(output_path)  #三种变形下都正确的比例