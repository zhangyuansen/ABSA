# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""InstructUIE Dataset."""

import json
import os
import random
import datasets
from hashlib import md5

logger = datasets.logging.get_logger(__name__)
TASK_CONFIG_FILES = {"train": "train_tasks.json", "dev": "dev_tasks.json", "test": "test_tasks.json"}
INSTRUCTION_STRATEGIES = ['single', 'multiple']
ANSWER_PREFIX = "Answer:"
SINGLE_QUOTES_SUBSTITUTE = "#$%#"
AUX_PROB = 0.3


def gen_cache_path(cache_dir, data_args):
    hash_str = data_args.data_dir + data_args.task_config_dir + \
               data_args.instruction_file + data_args.instruction_strategy + \
               str(data_args.max_num_instances_per_task) + str(data_args.max_num_instances_per_eval_task)
    hash_obj = md5(hash_str.encode("utf-8"))
    hash_id = hash_obj.hexdigest()
    cache_path = os.path.join(cache_dir, str(hash_id))

    return cache_path


def check_path(path):
    if not path or not os.path.exists(path):
        raise ValueError('{} is not valid, please check the input path!'.format(path))


def save_ds(instances, file_name):
    with open(file_name, "w+", encoding='utf-8') as fi:
        json.dump(instances, fi, ensure_ascii=False, indent=2)


class UIEConfig(datasets.BuilderConfig):
    """
    Config dataset load procedure.

    Args:
        data_dir: task data dir, which contains the corresponding dataset dirs
        prompt_path: prompt json file, which saves task and its prompts map
        task_file: task config file, save training and testing split config, and sampling strategies.
         Support two sampling strategies: 'random' indicates random sampling, while 'full' means to return all samples.
        max_num_instances_per_task: max training sample size of each task
        max_num_instances_per_eval_task: max eval sample size of each task
    """

    def __init__(
            self,
            *args,
            data_dir=None,
            instruction_file=None,
            instruction_strategy=None,
            task_config_dir=None,
            num_examples=None,
            max_num_instances_per_task=None,
            max_num_instances_per_eval_task=None,
            over_sampling=None,
            test_format="test",
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_examples = num_examples
        self.over_sampling = over_sampling
        self.test_format=test_format
        self.instructions = self._parse_instruction(instruction_file)
        self.task_configs = self._parse_task_config(task_config_dir)
        self.instruction_strategy = instruction_strategy
        self.max_num_instances_per_task = max_num_instances_per_task
        self.max_num_instances_per_eval_task = max_num_instances_per_eval_task

    def _parse_instruction(self, instruction_file):
        """
        Instruction example:
        {
          "RE": [
            {"instruction_type": "zero-shot", "instruction": "Given a phrase that describes the relationship between
            two words, extract the words and the lexical relationship between them.
            The output format should be :[(word1, relation, word2)]. \n"},
          ],
          "NER": [
            {"instruction_type": "zero-shot", "instruction": "Please list all entity words in the text that
            fit the category.Output format is [(word1, type1), (word2, type2))]. \n"},
          ],
          "EE": [
            {"instruction_type": "zero-shot", "instruction": "Extract the event information in the text
            and return them in the event list. \n"}
          ]
        }
        """
        if not instruction_file:
            return None
        instructions = {"zero-shot": {}, "few-shot": {}}

        with open(instruction_file, 'r+') as f:
            origin_instructions = json.load(f)

        for task in origin_instructions:
            for task_instruction in origin_instructions[task]:
                instruct_type = task_instruction["instruction_type"]
                if instruct_type == "zero-shot":
                    instructions['zero-shot'][task] = instructions['zero-shot'].get(task, [])
                    instructions['zero-shot'][task].append(task_instruction["instruction"])
                elif instruct_type == "few-shot":
                    instructions['few-shot'][task] = instructions['few-shot'].get(task, [])
                    instructions['few-shot'][task].append(task_instruction["instruction"])
                else:
                    raise ValueError("Invalid instruction type {}, please check your instruction file {}"
                                     .format(instruct_type, instruction_file))
        return instructions

    def _parse_task_config(self, task_config_dir):
        """
        Task config file example:
            {
              "RE": [
                {"sampling strategy": "random", "dataset name": "conll04"}
              ],
              "NER": [
                {"sampling strategy": "random", "dataset name": "ACE05_coarse-grained"},
                {"sampling strategy": "full", "dataset name": "conll2003"}
              ],
              "EE": [
                {"sampling strategy": "random", "dataset name": "GENIA"}
              ]
            }
        """
        if not task_config_dir:
            return None

        task_configs = {}
        for task, file_name in TASK_CONFIG_FILES.items():
            task_config_file = os.path.join(task_config_dir, file_name)

            if not os.path.exists(task_config_file):
                raise ValueError('Please check {} config, {} not exists!'.format(task, task_config_file))

            with open(task_config_file, 'r+') as f:
                task_configs[task] = json.loads(f.read())

        return task_configs


# TODO, few-shot, 需要 load 的时候就将值存好，放在 "Examples" 里面
class UIEInstructions(datasets.GeneratorBasedBuilder):
    """InstructUIE Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = UIEConfig
    BUILDER_CONFIGS = [
        UIEConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "Task": datasets.Value("string"),
                    "Dataset": datasets.Value("string"),
                    "subset": datasets.Value("string"),
                    "Samples": [{
                        "id": datasets.Value("string"),
                        "sentence": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "ground_truth": datasets.Value("string")
                    }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "sentence": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "instruction": datasets.Value("string"),
                        "ground_truth": datasets.Value("string")
                    }
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_configs is None:
            logger.error("Please provide right input: data_dir or task_config_dir!")

        # split dir save datasets
        # task config to specify train,dev,test
        split_dir = self.config.data_dir
        task_configs = self.config.task_configs
        
        test_dict={"test":"test","AddDiff":"AddDiff_test","RevTgt":"RevTgt_test","RevNon":"RevNon_test","Origin":"Origin_test"}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['train'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['dev'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['test'],
                    "max_num_instances_per_task": None,  # default load total test samples to test
                    "subset": test_dict[self.config.test_format]
                }),
        ]

    # def _load_dataset(self, dataset_path):
    #     with open(dataset_path, encoding="utf-8") as task_f:
    #         s = task_f.read()
    #         instances = json.loads(s)

    #     return instances
    def _load_dataset(self, dataset_path, labels_path):
        with open(dataset_path, encoding="utf-8") as task_f:
            s = task_f.read()
            instances = json.loads(s)
        with open(labels_path, encoding="utf-8") as labels_f:
            labels = json.load(labels_f)

        return instances, labels

    def _get_instruction(self, task):
        assert self.config.instruction_strategy in INSTRUCTION_STRATEGIES
        if self.config.num_examples is not None and self.config.num_examples > 0:
            task_instructions = self.config.instructions['few-shot'][task]
        else:
            task_instructions = self.config.instructions['zero-shot'][task]
        if self.config.instruction_strategy == "single":
            return task_instructions[0]
        else:
            return random.choice(task_instructions)

    def _sampling_dataset(self, instances,sampling_strategy, max_num_instances):
        # if sampling_strategy == 'random' and max_num_instances is not None and max_num_instances >= 0:
        #     instances = instances[:max_num_instances]
        neutral_num=0
        for key in instances.keys():
            term_lists=instances[key]['term_list']
            for term_key in term_lists.keys():
                if term_lists[term_key]['polarity']=='neutral':
                    neutral_num+=1
        
        if self.config.over_sampling:
            over_sampling_num=int(self.config.over_sampling)
            origin_instances = instances.copy()
            while neutral_num < over_sampling_num:
                keys=list(origin_instances.keys())
                random_instance_key=random.choice(keys)
                random_instance=origin_instances[random_instance_key]
                random_instance_term_lists=random_instance['term_list']
                for term_key in random_instance_term_lists.keys():
                    if random_instance_term_lists[term_key]['polarity']=='neutral':
                        new_term_lists={key:value for key,value in random_instance_term_lists.items() if key==term_key}
                        new_instance=random_instance.copy()
                        new_instance['term_list']=new_term_lists
                        index=1
                        while random_instance_key+"_{}".format(index) in instances.keys():
                            index+=1
                        new_index=random_instance_key+"_"+str(index)
                        instances[new_index]=new_instance
                        neutral_num+=1
        print("neutral num:",neutral_num)
                
        return instances
    
    def load_ABSA_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        print("!!!!!!!")
        print(dataset_path)
        instances, labels = self._load_dataset(dataset_path, labels_path)
        if subset=='train':
            instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)
        
        sample_template = {"Task":"ABSA", "Dataset": dataset_name, "Samples":[], "subset": subset}
        for idx,id in enumerate(instances.keys()):
            example = sample_template.copy()
            code_prompt=''
            sample=instances[id]
            sentence=sample['sentence']
            code_prompt += 'class ABSA:\n'
            
            code_prompt += '\t# The goal of this task is to classify the sentiment polarity (positive, negative, or neutral) expressed on an aspect extracted from a review sentence.\n'
            code_prompt += '\t# self.sentence is the sentence that contanins target aspect.\n'
            code_prompt += '\t# self.aspect is the target aspect term that we need to get its sentiment polarity.\n'
            code_prompt += '\t# self.aspect.polarity is the sentiment polarity attribute of the target aspect.\n'
            code_prompt += '\t# self.aspect.opinion_words are the basis for the sentiment polarity.\n'
            
            code_prompt += '\tdef __init__(self):\n'
            code_prompt += '\t\tself.sentence="{0}"\n'

            if 'test' not in subset: 
                aspect_terms=sample['term_list']
                aspects={}
                for aspect_key in aspect_terms:
                    aspect_term=aspect_terms[aspect_key]
                    term=aspect_term['term']
                    polarity=aspect_term['polarity']
                    opinion=aspect_term['opinion_words']
                    aspects[term]=[polarity,opinion]  #'battery life':['positive', ['good']]
                num=1
                for aspect in aspects:
                    code_prompt += '\t\tself.aspect{}="{}"\n'.format(num,aspect)
                    num+=1
                code_prompt += '\n'
                code_prompt += '\tdef get_the_sentiment_polarity_and_opinion_word(self):\n'


                label=''
                num=1
                for aspect in aspects:
                    label += '\t\tself.aspect{}.polarity="{}"\n'.format(num,aspects[aspect][0])
                    label += '\t\tself.aspect{}.opinion_words={}\n'.format(num,aspects[aspect][1])
                    num+=1
            else:
                aspect_term=sample['term']
                polarity=sample['polarity']
                code_prompt += '\t\tself.aspect1="{}"\n'.format(aspect_term)
                code_prompt += '\n'
                code_prompt += '\tdef get_the_sentiment_polarity_and_opinion_word(self):\n'
                label=''
                label += '\t\tself.aspect1.polarity="{}"\n'.format(polarity)
            example["Instance"] = {
                "id": id,
                "sentence": sentence,
                "label": label,
                "ground_truth": label,
                "instruction": code_prompt
            }

            yield example
            
    
    '''
    #codegen可行prompt
    def load_ABSA_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Task":"ABSA", "Dataset": dataset_name, "Samples":[], "subset": subset}

        for idx,id in enumerate(instances.keys()):
            instruction="The goal of this task is to classify the sentiment polarity expressed on an aspect extracted from a review sentence.\n"
            instruction+="Option:positive,negative,neutral\n"

            example = sample_template.copy()
            code_prompt=''
            sample=instances[id]
            sentence=sample['sentence']
            instruction+='The sentence is "{0}".\n'
            if 'test' not in subset:
                aspect_terms=sample['term_list']
                aspects={}
                for aspect_key in aspect_terms:
                    aspect_term=aspect_terms[aspect_key]
                    term=aspect_term['term']
                    polarity=aspect_term['polarity']
                    opinion=aspect_term['opinion_words']
                    aspects[term]=[polarity,opinion]  #'battery life':['positive', ['good']]
            else:
                aspects={}
                term=sample['term']
                polarity=sample['polarity']
                aspects[term]=[polarity]

            aspects_list=list(aspects.keys())
            instruction+='The aspects in the sentence are "{}".\n'.format(aspects_list)
            instruction+='Answer:'
                
            label = ", ".join("({}, {})".format(aspect, aspects[aspect][0]) for aspect in aspects)
            example["Instance"] = {
                "id": str(idx),
                "sentence": sentence,
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example
    '''
    '''
    def load_ABSA_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Task":"ABSA", "Dataset": dataset_name, "Samples":[], "subset": subset}

        for idx,id in enumerate(instances.keys()):
            instruction="The goal of this task is to classify the sentiment polarity expressed on an aspect extracted from a review sentence.The output type is (aspect, polarity)\n"
            instruction+='The sentence is "{0}". \n'
            instruction+="Option: positive, negative, neutral. \n"

            example = sample_template.copy()
            code_prompt=''
            sample=instances[id]
            sentence=sample['sentence']
            
            if 'test' not in subset:
                aspect_terms=sample['term_list']
                aspects={}
                for aspect_key in aspect_terms:
                    aspect_term=aspect_terms[aspect_key]
                    term=aspect_term['term']
                    polarity=aspect_term['polarity']
                    opinion=aspect_term['opinion_words']
                    aspects[term]=[polarity,opinion]  #'battery life':['positive', ['good']]
            else:
                aspects={}
                term=sample['term']
                polarity=sample['polarity']
                aspects[term]=[polarity]

            aspects_list=list(aspects.keys())
            instruction+='The aspects in the sentence are "{}".\n'.format(aspects_list)
            instruction+='Answer:'
                
            label = ", ".join(["aspect:{}, polarity:{}".format(aspect, aspects[aspect][0]) for aspect in aspects])
            example["Instance"] = {
                "id": id,
                "sentence": sentence,
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example
    '''
    def load_NER_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        # TODO, support few-shot
        sample_template = {"Task": "NER", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('NER')
            instruction += " Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + "Answer:"
            kv_pairs = []

            for entity in instance['entities']:
                if entity['type'] == 'NA' or entity['type'] == '':
                    continue
                kv_pair = [entity['name'], entity['type']]
                kv_pairs.append(kv_pair)

            if len(kv_pairs) > 0:
                label = ",".join([" ( {}, {})".format(k, v) for (k, v) in kv_pairs])
            else:
                label = " None"

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example
    def load_RE_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Task": "RE", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        # instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('RE')
            instruction += " Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + "Answer:"
            relation_pairs = []
            ground_truth_pairs = []

            for relation in instance['relations']:
                if relation['type'] == 'NA' or relation['type'] == '':
                    ground_truth_pairs.append([relation['head']['name'], 'NA', relation['tail']['name']])
                    continue
                relation_pair = [relation['head']['name'], relation['type'], relation['tail']['name']]
                ground_truth_pairs.append(relation_pair)
                relation_pairs.append(relation_pair)

            if len(relation_pairs) > 0:
                label = ",".join([" ( {}, {}, {})".format(h, r, t) for (h, r, t) in relation_pairs])
            else:
                label = ' None'

            if len(ground_truth_pairs) > 0:
                ground_truth = ",".join([" ( {}, {}, {})".format(h, r, t) for (h, r, t) in ground_truth_pairs])
            else:
                logger.error("******Error item: {}******".format(instance))
                raise Exception('Dataset Error:{}, No ground truth!'.format(dataset_name))

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": ground_truth,
                "instruction": instruction
            }

            yield example

    def load_EE_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Task": "EE", "Dataset": dataset_name, "Samples": [], "subset": subset}

        # TODO, reconstruct Event Instruction to two stage
        # TODO, check
        labels_str = f'Event type: {labels[0]}, Arguments type: {labels[1]}.'
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('EE')
            instruction += " Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + "Answer:"
            event_pairs = []

            for k, event in enumerate(instance['events']):
                instance['events'][k]['trigger'] = event['trigger'].replace("'", SINGLE_QUOTES_SUBSTITUTE)
                instance['events'][k]['type'] = event['type'].replace("'", SINGLE_QUOTES_SUBSTITUTE)

                if event['type'] == 'NA' or event['type'] == '':
                    continue
                event_type = event['type']
                event_trigger = event['trigger']
                event_arguments = ["(name:{},role:{})".format(argument['name'], argument['role']) for
                                   argument in event['arguments']]

                event_arguments = "None" if not event_arguments else ", ".join(event_arguments)
                event_pair = [event_type, event_trigger, event_arguments]
                event_pairs.append(event_pair)

            if len(event_pairs) > 0:
                label = ",".join([" ( type: {}, trigger: {}, arguments: {})".format(type, trigger, arguments)
                                   for (type, trigger, arguments) in event_pairs])
            else:
                label = ' None'

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example

    def _generate_examples(self, path=None, task_config=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")

        for task in task_config:
            if task == "NER":
                load_func = self.load_NER_dataset
            elif task == 'RE':
                load_func = self.load_RE_dataset
            elif task == 'EE':
                load_func = self.load_EE_dataset
            elif task == 'ES':
                load_func = self.load_ES_dataset
            elif task == 'ET':
                load_func = self.load_ET_dataset
            elif task == 'EP':
                load_func = self.load_EP_dataset
            elif task == 'EPR':
                load_func = self.load_EPR_dataset
            elif task == "ABSA":
                load_func = self.load_ABSA_dataset
            else:
                raise ValueError("Unsupport {} task, plz check {} task config!".format(task, subset))

            # load dataset
            for dataset in task_config[task]:
                ds_name = dataset["dataset name"]
                sampling_strategy = dataset.get("sampling strategy", "random")
                ds_path = os.path.join(path, task, ds_name, subset + '.json')
                labels_path = os.path.join(path, task, ds_name, 'labels.json')
                print(ds_path)

                assert os.path.exists(ds_path)

                idx = -1
                instances = []
                for sample in load_func(ds_path,labels_path, ds_name, sampling_strategy, max_num_instances_per_task,
                                        subset):
                    idx += 1
                    instances.append(sample)
                    yield f"{task}##{ds_path}##{idx}", sample
