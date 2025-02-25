import json
from tqdm import tqdm
import os
os.system('clear')
from util.model_util import SVRAG_InternVL2, SVRAG_Phi
from util.print_result import print_top_k_accuracy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='Phi3v',
                    choices=['Phi3v', 'InternVL2'], 
                    help="retriever name", 
                    type=str)
parser.add_argument("--k", default=1, help="top k image to use in QA", type=int)
args = parser.parse_args()


if __name__ == "__main__":
    
    if args.model == 'Phi3v':
        model = SVRAG_Phi(model_name='puar-playground/Col-Phi-3-V')
    elif args.model == 'InternVL2':
        model = SVRAG_InternVL2(model_name='puar-playground/Col-InternVL2-4B')

    # load test data
    test_data = json.load(open('slidevqa_dev.json', 'r'))
    data_point = test_data['aguidetoindonesiasnewgovernment-141104024232-conversion-gate02_95']
    
    # set input
    question_list = data_point['question']
    answer_list = data_point['answer']
    image_list = data_point['image_urls']

    true_indicies = data_point['index_list']

    # Do retrieval
    _, retrieved_indicies = model.retrieve(query_list=question_list, image_list=image_list)
    retrieved_indicies = retrieved_indicies.tolist()

    # evaluate retrieval
    print('Doing retrieval')
    acc_str = print_top_k_accuracy([[true_indicies, retrieved_indicies]], k_list=[1, 5])
    print(acc_str)

    # generate answer
    print('Doing question answering')
    model.disable_lora_if_present()
    for question, answer, retrieved_index in zip(question_list, answer_list, retrieved_indicies):

        # extract top images
        retrieved_index_top = retrieved_index[:args.k]
        true_img_list = [x for i, x in enumerate(image_list) if i in retrieved_index_top]

        # generate
        answer_model = model.ask(question, true_img_list)

        print('>' * 120)
        print(question)
        print(f'answer_true: {answer}')
        print(f'answer_model: {answer_model}')

