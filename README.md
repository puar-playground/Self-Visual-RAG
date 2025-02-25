# SV-RAG
This is an implementation of the Col-retriever model in the [SV-RAG](https://arxiv.org/abs/2411.01106) paper, adapted from the [ColPali](https://github.com/illuin-tech/colpali) repository. This project integrates two powerful base models: 
- [Phi-3-V](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) (Microsoft)
- [InternVL2](https://huggingface.co/OpenGVLab/InternVL2-4B). (OpenGVLab)

These models are fine-tuned with LoRA adapters for document retrieval task.

## Inference
We trained two models using LoRA: [Col-Phi-3-V](https://huggingface.co/puar-playground/Col-Phi-3-V) and [Col-InternVL2](https://huggingface.co/puar-playground/Col-InternVL2-4B). 
You can test the retrieval model using the `run_test.py` script with demo data (demo_data/slidevqa_dev.json):
```
python run_test.py --model ColInternVL2
python run_test.py --model ColPhi
```
This script will demonstrate retrieval performance on sample query data.

## 📖 Reference
```
@article{chen2024lora,
  title={LoRA-Contextualizing Adaptation of Large Multimodal Models for Long Document Understanding},
  author={Chen, Jian and Zhang, Ruiyi and Zhou, Yufan and Yu, Tong and Dernoncourt, Franck and Gu, Jiuxiang and Rossi, Ryan A and Chen, Changyou and Sun, Tong},
  journal={arXiv preprint arXiv:2411.01106},
  year={2024}
}
```

