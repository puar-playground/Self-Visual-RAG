import json
from tqdm import tqdm
import os
os.system('clear')
from util.retrieval_model_util import ColPhiRetriever, ColInternVL2Retriever
from util.print_result import print_top_k_accuracy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='ColInternVL2',
                    choices=['ColPhi', 'ColInternVL2'], 
                    help="retriever name", 
                    type=str)
args = parser.parse_args()


if __name__ == "__main__":
    
    if args.model == 'ColPhi':
        model = ColPhiRetriever(model_name='puar-playground/Col-Phi-3-V')
    elif args.model == 'ColInternVL2':
        model = ColInternVL2Retriever(model_name='puar-playground/Col-InternVL2-4B')

    test_data = json.load(open('./demo_data/slidevqa_dev.json', 'r'))
    
    results = []
    pbar = tqdm(list(test_data.keys()), ncols=120)
    for doc_id in pbar:

        data_point = test_data[doc_id]
        question_list = data_point['question']
        image_list = data_point['image_urls']
        true_indicies = data_point['index_list']

        _, retrieved_indicies = model.retrieve(query_list=question_list, image_list=image_list)
        retrieved_indicies = retrieved_indicies.tolist()

        results.append([true_indicies, retrieved_indicies])

        acc_str = print_top_k_accuracy(results, k_list=[1, 5])
        pbar.set_description(acc_str)
    


