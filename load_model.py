import os
import json
from collections import namedtuple

import tensorflow as tf

CONFIG_JSON = 'config.json'
FILE_KEY_SUFFIX = '_file'

ModelBox = namedtuple('ModelBox', ('inputs', 'outputs', 'vocab_file', 'supersense_vocab_file', 'do_lower_case'))


def load_config(import_path):
    with tf.io.gfile.GFile(os.path.join(import_path, CONFIG_JSON)) as f:
        config = json.load(f)

    file_keys = [key for key in config.keys() if key.endswith(FILE_KEY_SUFFIX)]

    for key in file_keys:
        config[key] = os.path.join(import_path, config[key])
    return config


def load(import_path, import_scope=None, session=None):
    if session is None:
        session = tf.get_default_session()

    graph = session.graph
    export_dir = os.path.dirname((tf.io.gfile.glob(os.path.join(import_path, 'variables')) + tf.io.gfile.glob(os.path.join(import_path, '**', 'variables')))[0])
    model = tf.saved_model.load(export_dir=export_dir, sess=session, tags=[tf.saved_model.SERVING],
                                import_scope=import_scope)
    serve_def = model.signature_def[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    def with_scope(name):
        if import_scope:
            return os.path.join(import_scope, name)
        return name

    inputs, outputs = ({key: graph.get_tensor_by_name(with_scope(info.name)) for key, info in puts.items()}
                       for puts in (serve_def.inputs, serve_def.outputs))
    config = load_config(import_path)

    vocab_file = config['vocab_file']
    supersense_vocab_file = config['supersense_vocab_file']
    do_lower_case = config['do_lower_case'] if 'do_lower_case' in config else True

    return ModelBox(inputs=inputs, outputs=outputs,
                    vocab_file=vocab_file, supersense_vocab_file=supersense_vocab_file, do_lower_case=do_lower_case)







