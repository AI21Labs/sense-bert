# SenseBERT: Driving Some Sense into BERT

This is the code for loading the SenseBERT model, described in [our paper](https://www.aclweb.org/anthology/2020.acl-main.423.pdf) from ACL 2020.

## Available models

We made two SenseBERT models public:

* sensebert-base-uncased
* sensebert-large-uncased

These models have the same number of parameters as Google's BERT models, except for the following (both changes are described in our paper thoroughly):
1. We use a larger vocabulary.
2. We add a supersense prediction head. The sense embeddings are also used as inputs to the model. 

## Requirements

* Python 3.7 or higher
* TensorFlow 1.15
* NLTK

You can install these using:
```bash
pip install -r requirements.txt
```

## Usage


### Supersense and MLM predictions

This is an example for making Masked Language Modeling (MLM) and supersense predictions based on SenseBERT:
```python
import tensorflow as tf
from sensebert import SenseBert

with tf.Session() as session:
    sensebert_model = SenseBert("sensebert-base-uncased", session=session)  # or sensebert-large-uncased
    input_ids, input_mask = sensebert_model.tokenize(["I went to the store to buy some groceries.", "The store was closed."])
    model_outputs = sensebert_model.run(input_ids, input_mask)

contextualized_embeddings, mlm_logits, supersense_logits = model_outputs  # these are NumPy arrays
```
Note that both vocabularies (tokens and supersenses) are available for you via ```sensebert_model.tokenizer```. For example, in order to predict the supersense of the word 'groceries' in the above example, you may run
```python
import numpy as np

print(sensebert_model.tokenizer.convert_ids_to_senses([np.argmax(supersense_logits[0][9])]))
```
This will output:
```
['noun.artifact']
```
### Fine-tuning

If you want to fine-tune SenseBERT, run
```python
sensebert_model = SenseBert("sensebert-base-uncased", session=session)  # or sensebert-large-uncased
```

```sensebert_model.model.input_ids``` and ```sensebert_model.model.input_mask``` are the model's tensorflow placeholders, and ```sensebert_model.model.contextualized_embeddings```, ```sensebert_model.model.mlm_logits``` and ```sensebert_model.model.supersense_logits``` are the output tensors. You can take any of these three tensors and build your graph on top of them. 


### Download SenseBERT to your local machine

In order to avoid high latency, we recommend to download the model once to your local machine. Our code also supports initializations from local directories. 
For that, you will need to install ```gsutil```. Once you have it, run one of the following
```shell script
gsutil -m cp -r gs://ai21-public-models/sensebert-base-uncased PATH/TO/DIR
gsutil -m cp -r gs://ai21-public-models/sensebert-large-uncased PATH/TO/DIR
```

Then you can go ahead and use our code exactly as before, with
```python
sensebert_model = SenseBert("PATH/TO/DIR", session=session)
```

## Citation 
If you use our model for your research, please cite our paper:

 ```
@inproceedings{levine-etal-2020-sensebert,
    title = "{S}ense{BERT}: Driving Some Sense into {BERT}",
    author = "Levine, Yoav  and
      Lenz, Barak  and
      Dagan, Or  and
      Ram, Ori  and
      Padnos, Dan  and
      Sharir, Or  and
      Shalev-Shwartz, Shai  and
      Shashua, Amnon  and
      Shoham, Yoav",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.423",
    pages = "4656--4667",
}
```

