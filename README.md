# Named Entity Recognition with DocRED and DistilBERT
## Overview
This project focuses on applying the DistilBERT transformer (encoder) model to the task of Named Entity Recognition (NER) using the DocRED dataset.
The project consists of two Jupyter Notebook files:
1. **create_hf_dataset.ipynb** - Here the processing of the DocRED dataset is happening. The data is prepared into the right form for fine-tuning and pushed to Huggingface Hub.
2. **finetune-distilbert.ipynb** - In this notebook the finetuning and evaluation of the model is done. Also at end, the model gets pushed to the Huggingface Hub.

## Dataset
DocRED is a dataset which contains sentences with named entities and the relation between them.
For this project only the named entities are used in order to classify them with the final model.

The dataset is structured as follows:
```json
{
  "title": "string",
  "sents": [
    ["word in sent 0"],
    ["word in sent 1"]
  ],
  "vertexSet": [
    [
      {
        "name": "mention_name",
        "sent_id": "mention in which sentence",
        "pos": "position of mention in a sentence",
        "type": "NER_type"
      },
      {"another mention"}
    ],
    ["another entity"]
  ],
  "labels": [
    {
      "h": "idx of head entity in vertexSet",
      "t": "idx of tail entity in vertexSet",
      "r": "relation",
      "evidence": "evidence sentences' id"
    }
  ]
}

```

More information about the DocRED dataset can be found here: 
- Github Repo: https://github.com/thunlp/DocRED
- Paper: https://arxiv.org/abs/1906.06127v3

## Model
As the model for finetuning DistilBERT is used. DistilBERT is an encoder-based model with 66 million parameters that is a distilled version of the larger BERT model. It is faster, lighter and more cost-effective to train as it has less parameters. This makes DistilBERT suitable for this project as the model is finetuned on a free tier of Google Colab that offers a T4 GPU.
A detailed description of DistilBERT can be found here: https://huggingface.co/docs/transformers/model_doc/distilbert

## Performance Evaluation
The model's performance across different entity categories is summarized in the following table:

| Entity Type | Precision | Recall | F1-Score | Number of Instances |
|-------------|-----------|--------|----------|---------------------|
| LOC         | 0.9130    | 0.9298 | 0.9214   | 8007                |
| MISC        | 0.7768    | 0.7922 | 0.7844   | 3854                |
| NUM         | 0.8916    | 0.9238 | 0.9074   | 1300                |
| ORG         | 0.8309    | 0.8538 | 0.8422   | 3811                |
| PER         | 0.9601    | 0.9647 | 0.9624   | 4709                |
| TIME        | 0.9402    | 0.9558 | 0.9479   | 3961                |

Overall model performance:

- Precision: 0.8918
- Recall: 0.9080
- F1-Score: 0.8998
- Accuracy: 0.9767

The confusion matrix of the finetuned model:
![image](https://github.com/dennis-hauser/transformer-finetuning-ner/assets/104132152/0c4059d9-007f-4feb-bf7a-992c49c73240)







DocRED:
Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin, Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou, Maosong Sun. "DocRED: A Large-Scale Document-Level Relation Extraction Dataset". In Proceedings of ACL 2019, 2019.
