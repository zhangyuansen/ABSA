# import json
# with open("/workspace/output/t5-700M-demo-without_dataset_name/predict_eval_predictions.jsonl","r") as f:
#     json.loads(f)
#     print(f)
    
import json_lines
with open('/workspace/output/t5-700M-demo-without_dataset_name/predict_eval_predictions.jsonl', 'rb') as f: 
   for item in json_lines.reader(f):
    #    print(item.keys())
       if item['Instance']['ground_truth']==item['Prediction']:
           print(item)
       

