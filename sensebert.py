import os
from collections import namedtuple
import numpy as np
import tensorflow as tf

from tokenization import FullTokenizer

SenseBertGraph = namedtuple('SenseBertGraph',
                            ('input_ids', 'input_mask', 'contextualized_embeddings', 'mlm_logits', 'supersense_losits'))

model_paths = {
    'sensebert-base-uncased': 'gs://ai21-public-models/sensebert-base-uncased',
    'sensebert-large-uncased': 'gs://ai21-public-models/sensebert-large-uncased'
}
contextualized_embeddings_tensor_name = "bert/encoder/Reshape_13:0"


def load_model(name_or_path, session=None):
    if name_or_path in model_paths:
        print(f"Loading the known model '{name_or_path}'")
        model_path = model_paths[name_or_path]
    else:
        print(f"This is not a known model. Assuming this is a path or a url...")
        model_path = name_or_path

    if session is None:
        session = tf.get_default_session()

    graph = session.graph
    export_dir = os.path.dirname((tf.io.gfile.glob(os.path.join(model_path, 'variables')) +
                                  tf.io.gfile.glob(os.path.join(model_path, '**', 'variables')))[0])
    model = tf.saved_model.load(export_dir=export_dir, sess=session, tags=[tf.saved_model.SERVING])
    serve_def = model.signature_def[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    inputs, outputs = ({key: graph.get_tensor_by_name(info.name) for key, info in puts.items()}
                       for puts in (serve_def.inputs, serve_def.outputs))
    outputs['contextualized_embeddings'] = session.graph.get_tensor_by_name(contextualized_embeddings_tensor_name)

    vocab_file = os.path.join(model_path, "vocab.txt")
    supersense_vocab_file = os.path.join(model_path, "supersense_vocab.txt")
    tokenizer = FullTokenizer(vocab_file, senses_file=supersense_vocab_file)

    input_ids, input_mask = model.inputs['input_ids'], model.inputs['input_mask']
    embeddings, mlm, ss = model.outputs['contextualized_embeddings'], model.outputs['masked_lm'], model.outputs['ss']

    return SenseBertGraph(input_ids=input_ids, input_mask=input_mask,
                          contextualized_embeddings=embeddings, supersense_losits=ss, mlm_logits=mlm),  tokenizer


class SenseBert:
    def __init__(self, name_or_path, max_seq_length=512, session=None):
        self.name_or_path = name_or_path
        self.max_seq_length = max_seq_length
        self.session = session if session else tf.get_default_session()

        self.model, self.tokenizer = load_model(self.name_or_path, session=self.session)
        self.model_output_tensors = [self.model.contextualized_embeddings,
                                     self.model.mlm_logits, self.model.supersense_losits]

        self.start_sym = self.tokenizer.start_sym
        self.end_sym = self.tokenizer.end_sym
        self.pad_sym_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_sym])[0]

    def tokenize(self, inputs):
        """
        Gets a string or a list of strings, and returns a tuple (input_ids, input_mask) to use as inputs for SenseBERT.
        Both share the same shape: [batch_size, sequence_length]
        :param inputs: A string or a list of string
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        assert isinstance(inputs, list)
        assert all([isinstance(s, str) for s in inputs])

        # Tokenizing all inputs
        all_token_ids = []
        for inp in inputs:
            tokens = [self.start_sym] + self.tokenizer.tokenize(inp)[0] + [self.end_sym]
            assert len(tokens) <= self.max_seq_length

            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            all_token_ids.append(tokens)

        # Deciding the maximum sequence length and padding accordingly
        max_len = np.max([len(token_ids) for token_ids in all_token_ids])
        input_ids, input_mask = [], []
        for token_ids in all_token_ids:
            to_pad = max_len - len(token_ids)
            input_ids.append(token_ids + [self.pad_sym_id] * to_pad)
            input_mask.append([1] * len(token_ids) + [0] * to_pad)

        return input_ids, input_mask

    def run(self, input_ids, input_mask):
        model_outputs = self.session.run(self.model_output_tensors,
                                         feed_dict={self.model.input_ids: input_ids,
                                                    self.model.input_mask: input_mask})

        return model_outputs
