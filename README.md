# Self-Visual-RAG
This repository provides an unofficial implementation of the Col-retriever model, developed based on the methodology presented in the [**SV-RAG**](https://arxiv.org/abs/2411.01106)**: LoRA-Contextualizing Adaptation of MLLMs for Long Document Understanding** paper, adapted from the [ColPali](https://github.com/illuin-tech/colpali) repository. This project integrates two powerful base models: 
- [Phi-3-V](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) (Microsoft)
- [InternVL2](https://huggingface.co/OpenGVLab/InternVL2-4B). (OpenGVLab)

These models are fine-tuned with LoRA adapters for document retrieval task.
![image](overview.png)
## Installation
```
conda create -n svrag python=3.10 -y
conda activate svrag
pip install -e .
pip install -r requirements.txt
```

## Retrieval Inference
We trained two models using LoRA: [Col-Phi-3-V](https://huggingface.co/puar-playground/Col-Phi-3-V) and [Col-InternVL2](https://huggingface.co/puar-playground/Col-InternVL2-4B). 
You can test the retrieval model using the `run_test.py` script with demo data [slidevqa_dev.json](https://github.com/puar-playground/SV-RAG/blob/main/demo_data/slidevqa_dev.json):
```
python test_retrieval --model InternVL2
python test_retrieval --model Phi
```

## Self-Visual-RAG Inference
These script demonstrate the full Self-Visual-RAG pipeline on demo query data.
```
python test_sv_rag --model InternVL2 --k 5
python test_sv_rag --model Phi
```
For the InternVL2 backbone, the VLLM supports multiple image inputs. You can use the `--k` flag to control the top-k retrieved images used for answer generation. However, increasing k will significantly increase memory consumption.

## Training
Col-InterVL2-4B model:
```
torchrun --nproc_per_node=8 --master_port=20001 scripts/train/train_colbert.py train_colInternVL2_4b_model.yaml
```
Col-Phi-3-V model:
```
torchrun --nproc_per_node=8 --master_port=20001 scripts/train/train_colbert.py train_colphi_model.yaml
```

## 📖 Reference
```
@article{chen2024lora,
  title={LoRA-Contextualizing Adaptation of Large Multimodal Models for Long Document Understanding},
  author={Chen, Jian and Zhang, Ruiyi and Zhou, Yufan and Yu, Tong and Dernoncourt, Franck and Gu, Jiuxiang and Rossi, Ryan A and Chen, Changyou and Sun, Tong},
  journal={arXiv preprint arXiv:2411.01106},
  year={2024}
}
```

