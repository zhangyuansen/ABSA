
import os
import json
import re


def judge_true(data):
    ground_truth=data['Instance']["ground_truth"]

    if "positive" in ground_truth:
        polarity="positive"
    elif "negative" in ground_truth:
        polarity="negative"
    else:
        polarity="neutral"
        
            
    predict=data["Prediction"]

    if "positive" in predict:
        predict_polarity="positive"
    elif "negative" in predict:
        predict_polarity="negative"
    else:
        predict_polarity="neutral"
    if polarity==predict_polarity:
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
            total_aspect+=1
            data=json.loads(line)
            if judge_true(data):
                total_truth+=1
    print(total_aspect)
    print(total_truth/total_aspect)

def caculate_acc(test_dict,name):
    total_aspect=0
    total_truth=0
    yiluo=0
    
    for key in test_dict.keys():
        total_aspect+=1
        data=test_dict[key]
        if judge_true(data):
            total_truth+=1

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
         
    



if __name__ == '__main__':
    output_path="/workspace/output/flan-t5-large-ABSA-laptop-all-annotion"
    caculate_all_acc(output_path)  #变形后的test set的acc
    caculate_acc_sep(output_path)   #计算每种变形下的ACC变化
    caculate_overall_acc(output_path)  #三种变形下都正确的比例