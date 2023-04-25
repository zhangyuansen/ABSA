from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,  # add
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )

class API:
    def __init__(self):
        self.output_dir="/workspace/output/t5-700M-demo-without_dataset_name"
        self.model=AutoModelForSeq2SeqLM.from_pretrained(self.output_dir)
        self.tokenizer=AutoTokenizer.from_pretrained(self.output_dir)

    def api(self,prompt):
        input_ids=self.tokenizer(prompt, return_tensors='pt').input_ids
        outputs=self.model.generate(input_ids,max_length=128)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)




if __name__ == "__main__":
    api=API()
    while True:
        user_input = input("Please enter your input to the model, or enter 'quit' to exit: ")
        if user_input.lower() == "quit":
            break
        else:
            # result=api.api("Task:RE\nDataset:New-York-Times-RE\nGiven a phrase that describes the relationship between two words, extract the words and the lexical relationship between them. The output format should be ( word1, relation, word2).\nOption: ethnicity, place lived, geographic distribution, company industry, country of administrative divisions, administrative division of country, location contains, person of company, profession, ethnicity of people, company shareholder among major shareholders, sports team of location, religion, neighborhood of, company major shareholders, place of death, nationality, children, company founders, company founded place, country of capital, company advisors, sports team location of teams, place of birth\n Text: 9 P.M. -LRB- Sundance -RRB- ONE PUNK UNDER GOD Jay Bakker , the pierced and tattooed preacher son of Jim Bakker and Tammy Faye Messner , the founders of the Praise the Lord Club , goes on a soul-searching mission to define his feelings about homosexuality , while visiting his ill mother and reconnecting with his father .\nAnswer:")
            result=api.api(user_input)
            print(result)
