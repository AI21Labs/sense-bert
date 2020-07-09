import os
from collections import namedtuple
import tensorflow as tf

from tokenization import FullTokenizer

_SenseBertGraph = namedtuple(
    'SenseBertGraph',
    ('input_ids', 'input_mask', 'contextualized_embeddings', 'mlm_logits', 'supersense_losits')
)

_MODEL_PATHS = {
    'sensebert-base-uncased': 'gs://ai21-public-models/sensebert-base-uncased',
    'sensebert-large-uncased': 'gs://ai21-public-models/sensebert-large-uncased'
}
_CONTEXTUALIZED_EMBEDDINGS_TENSOR_NAME = "bert/encoder/Reshape_13:0"


def _get_model_path(name_or_path):
    if name_or_path in _MODEL_PATHS:
        print(f"Loading the known model '{name_or_path}'")
        model_path = _MODEL_PATHS[name_or_path]
    else:
        print(f"This is not a known model. Assuming this is a path or a url...")
        model_path = name_or_path
    return model_path


def load_tokenizer(name_or_path):
    model_path = _get_model_path(name_or_path)
    vocab_file = os.path.join(model_path, "vocab.txt")
    supersense_vocab_file = os.path.join(model_path, "supersense_vocab.txt")
    return FullTokenizer(vocab_file=vocab_file, senses_file=supersense_vocab_file)


def _load_model(name_or_path, session=None):
    model_path = _get_model_path(name_or_path)

    if session is None:
        session = tf.get_default_session()

    model = tf.saved_model.load(export_dir=model_path, sess=session, tags=[tf.saved_model.SERVING])
    serve_def = model.signature_def[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    inputs, outputs = ({key: session.graph.get_tensor_by_name(info.name) for key, info in puts.items()}
                       for puts in (serve_def.inputs, serve_def.outputs))

    return _SenseBertGraph(
        input_ids=inputs['input_ids'],
        input_mask=inputs['input_mask'],
        contextualized_embeddings=session.graph.get_tensor_by_name(_CONTEXTUALIZED_EMBEDDINGS_TENSOR_NAME),
        supersense_losits=outputs['ss'],
        mlm_logits=outputs['masked_lm']
    )


class SenseBert:
    def __init__(self, name_or_path, max_seq_length=512, session=None):
        self.max_seq_length = max_seq_length
        self.session = session if session else tf.get_default_session()
        self.model = _load_model(name_or_path, session=self.session)
        self.tokenizer = load_tokenizer(name_or_path)

    def tokenize(self, inputs):
        """
        Gets a string or a list of strings, and returns a tuple (input_ids, input_mask) to use as inputs for SenseBERT.
        Both share the same shape: [batch_size, sequence_length] where sequence_length is the maximal sequence length.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        # tokenizing all inputs
        all_token_ids = []
        for inp in inputs:
            tokens = [self.tokenizer.start_sym] + self.tokenizer.tokenize(inp)[0] + [self.tokenizer.end_sym]
            assert len(tokens) <= self.max_seq_length
            all_token_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))

        # decide the maximum sequence length and pad accordingly
        max_len = max([len(token_ids) for token_ids in all_token_ids])
        input_ids, input_mask = [], []
        pad_sym_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_sym])
        for token_ids in all_token_ids:
            to_pad = max_len - len(token_ids)
            input_ids.append(token_ids + pad_sym_id * to_pad)
            input_mask.append([1] * len(token_ids) + [0] * to_pad)

        return input_ids, input_mask

    def run(self, input_ids, input_mask):
        return self.session.run(
            [self.model.contextualized_embeddings, self.model.mlm_logits, self.model.supersense_losits],
            feed_dict={self.model.input_ids: input_ids, self.model.input_mask: input_mask}
        )
