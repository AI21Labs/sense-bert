import numpy as np
import tensorflow as tf


from tokenization import FullTokenizer
from load_model import load


class SenseBERT:
    def __init__(self, import_path, session=None):
        self.import_path = import_path
        self.session = session if session else tf.get_default_session()

        self.model_box = load(self.import_path, session=self.session)
        self.tokenizer = FullTokenizer(self.model_box.vocab_file, senses_file=self.model_box.supersense_vocab_file,
                                       do_lower_case=self.model_box.do_lower_case)

        self.max_seq_length = 512
        self.input_ids_placeholder = self.model_box.inputs['input_ids']
        self.input_mask_placeholder = self.model_box.inputs['input_mask']

        self.model_output_tensors = [self.model_box.outputs['ss'], self.model_box.outputs['masked_lm']]

        self.start_sym = self.tokenizer.start_sym
        self.end_sym = self.tokenizer.end_sym
        self.pad_sym_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_sym])[0]

    def run(self, inputs):
        """
        Gets a list of strings or a single string, and returns SenseBERT's outputs
        :param inputs:
        :return:
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

        # Deciding the sequence length and padding accordingly
        max_len = np.max([len(token_ids) for token_ids in all_token_ids])
        input_ids, input_mask = [], []
        for token_ids in all_token_ids:
            to_pad = max_len - len(token_ids)
            input_ids.append(token_ids + [self.pad_sym_id] * to_pad)
            input_mask.append([1] * len(token_ids) + [0] * to_pad)

        # Running the model
        model_outputs = self.session.run(self.model_output_tensors,
                                         feed_dict={self.input_ids_placeholder: input_ids,
                                                    self.input_mask_placeholder: input_mask})

        return model_outputs



