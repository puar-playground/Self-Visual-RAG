# SV-RAG
This is an implementation of the Col-retriever model based on the source code of [ColPali)](https://github.com/illuin-tech/colpali) for base model: [Phi-3-V](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) and [InternVL2](https://huggingface.co/OpenGVLab/InternVL2-4B). 

## Inference
We trained two models using LoRA: [Col-Phi-3-V](https://huggingface.co/puar-playground/Col-Phi-3-V) and [Col-InternVL2](https://huggingface.co/puar-playground/Col-InternVL2-4B). The `run_test.py` provides a simple example for retrieval.
```
python run_test.py --model ColInternVL2
python run_test.py --model ColPhi
```
The script will demonstrate the retrieval model using demo data `demo_data/slidevqa_dev.json`.

## Reference
```
@article{chen2024lora,
  title={LoRA-Contextualizing Adaptation of Large Multimodal Models for Long Document Understanding},
  author={Chen, Jian and Zhang, Ruiyi and Zhou, Yufan and Yu, Tong and Dernoncourt, Franck and Gu, Jiuxiang and Rossi, Ryan A and Chen, Changyou and Sun, Tong},
  journal={arXiv preprint arXiv:2411.01106},
  year={2024}
}
```

