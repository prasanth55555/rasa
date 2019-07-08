import io
import logging
import numpy as np
import os
import pickle
import typing
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa.nlu.classifiers import INTENT_RANKING_LENGTH
from rasa.nlu.components import Component
from rasa.utils.common import is_logging_disabled
from rasa.nlu.classifiers.embedding_intent_classifier import EmbeddingIntentClassifier

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from tensorflow import Graph, Session, Tensor
    from rasa.nlu.config import RasaNLUModelConfig
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.model import Metadata
    from rasa.nlu.training_data import Message

try:
    import tensorflow as tf

    # avoid warning println on contrib import - remove for tf 2
    tf.contrib._warning = None
except ImportError:
    tf = None

class ResponseSelector(EmbeddingIntentClassifier):
    """Intent classifier using supervised embeddings.

        The embedding intent classifier embeds user inputs
        and intent labels into the same space.
        Supervised embeddings are trained by maximizing similarity between them.
        It also provides rankings of the labels that did not "win".

        The embedding intent classifier needs to be preceded by
        a featurizer in the pipeline.
        This featurizer creates the features used for the embeddings.
        It is recommended to use ``CountVectorsFeaturizer`` that
        can be optionally preceded by ``SpacyNLP`` and ``SpacyTokenizer``.

        Based on the starspace idea from: https://arxiv.org/abs/1709.03856.
        However, in this implementation the `mu` parameter is treated differently
        and additional hidden layers are added together with dropout.
        """

    provides = ["response", "response_ranking"]

    requires = ["text_features"]

    name = 'response_selector'

    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_a": [256, 128],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_b": [],
        # training parameters
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        "batch_size": [64, 256],
        # number of epochs
        "epochs": 300,
        # embedding parameters
        # dimension size of embedding vectors
        "embed_dim": 20,
        # how similar the algorithm should try
        # to make embedding vectors for correct intent labels
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect intent labels
        "mu_neg": -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the type of the similarity
        "similarity_type": "cosine",  # string 'cosine' or 'inner'
        # the number of incorrect intents, the algorithm will minimize
        # their similarity to the input words during training
        "num_neg": 20,
        # flag: if true, only minimize the maximum similarity for
        # incorrect intent labels
        "use_max_sim_neg": True,
        # set random seed to any int to get reproducible results
        # try to change to another int if you are not getting good results
        "random_seed": None,
        # regularization parameters
        # the scale of L2 regularization
        "C2": 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different intent labels
        "C_emb": 0.8,
        # dropout rate for rnn
        "droprate": 0.2,
        # flag: if true, the algorithm will split the intent labels into tokens
        #       and use bag-of-words representations for them
        "intent_tokenization_flag": False,
        # delimiter string to split the intent labels
        "intent_split_symbol": "_",
        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 10,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 1000,  # large values may hurt performance
    }

    def __init__(
            self,
            component_config: Optional[Dict[Text, Any]] = None,
            inv_response_dict: Optional[Dict[int, Text]] = None,
            encoded_all_responses: Optional[np.ndarray] = None,
            session: Optional["Session"] = None,
            graph: Optional["Graph"] = None,
            message_placeholder: Optional["Tensor"] = None,
            response_placeholder: Optional["Tensor"] = None,
            similarity_op: Optional["Tensor"] = None,
            word_embed: Optional["Tensor"] = None,
            response_embed: Optional["Tensor"] = None,
    ) -> None:
        super(ResponseSelector, self).__init__(component_config, inv_response_dict, encoded_all_responses,
                                               session, graph, message_placeholder, response_placeholder,
                                               similarity_op, word_embed, response_embed)


    def process(self, message: "Message", **kwargs: Any) -> None:
        """Return the most likely response and its similarity to the input."""

        response = {"name": None, "confidence": 0.0}
        response_ranking = []

        if self.session is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )

        else:
            # get features (bag of words) for a message
            # noinspection PyPep8Naming
            X = message.get("text_features").reshape(1, -1)

            # stack encoded_all_intents on top of each other
            # to create candidates for test examples
            # noinspection PyPep8Naming
            all_Y = self._create_all_Y(X.shape[0])

            # load tf graph and session
            response_ids, message_sim = self._calculate_message_sim(X, all_Y)

            # if X contains all zeros do not predict some label
            if X.any() and response_ids.size > 0:
                response = {
                    "name": self.inv_intent_dict[response_ids[0]],
                    "confidence": message_sim[0],
                }

                ranking = list(zip(list(response_ids), message_sim))
                ranking = ranking[:INTENT_RANKING_LENGTH]
                response_ranking = [
                    {"name": self.inv_intent_dict[intent_idx], "confidence": score}
                    for intent_idx, score in ranking
                ]

        message.set("response", response, add_to_output=True)
        message.set("response_ranking", response_ranking, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        if self.session is None:
            return {"file": None}

        checkpoint = os.path.join(model_dir, file_name + ".ckpt")

        try:
            os.makedirs(os.path.dirname(checkpoint))
        except OSError as e:
            # be happy if someone already created the path
            import errno

            if e.errno != errno.EEXIST:
                raise
        with self.graph.as_default():
            self.graph.clear_collection("message_placeholder")
            self.graph.add_to_collection("message_placeholder", self.a_in)

            self.graph.clear_collection("intent_placeholder")
            self.graph.add_to_collection("intent_placeholder", self.b_in)

            self.graph.clear_collection("similarity_op")
            self.graph.add_to_collection("similarity_op", self.sim_op)

            self.graph.clear_collection("word_embed")
            self.graph.add_to_collection("word_embed", self.word_embed)
            self.graph.clear_collection("intent_embed")
            self.graph.add_to_collection("intent_embed", self.intent_embed)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with io.open(
            os.path.join(model_dir, file_name + "_inv_response_dict.pkl"), "wb"
        ) as f:
            pickle.dump(self.inv_intent_dict, f)
        with io.open(
            os.path.join(model_dir, file_name + "_encoded_all_responses.pkl"), "wb"
        ) as f:
            pickle.dump(self.encoded_all_intents, f)

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: "Metadata" = None,
        cached_component: Optional["ResponseSelector"] = None,
        **kwargs: Any
    ) -> "ResponseSelector":

        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            checkpoint = os.path.join(model_dir, file_name + ".ckpt")
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()
                saver = tf.train.import_meta_graph(checkpoint + ".meta")

                saver.restore(sess, checkpoint)

                a_in = tf.get_collection("message_placeholder")[0]
                b_in = tf.get_collection("intent_placeholder")[0]

                sim_op = tf.get_collection("similarity_op")[0]

                word_embed = tf.get_collection("word_embed")[0]
                intent_embed = tf.get_collection("intent_embed")[0]

            with io.open(
                os.path.join(model_dir, file_name + "_inv_response_dict.pkl"), "rb"
            ) as f:
                inv_intent_dict = pickle.load(f)
            with io.open(
                os.path.join(model_dir, file_name + "_encoded_all_responses.pkl"), "rb"
            ) as f:
                encoded_all_intents = pickle.load(f)

            return cls(
                component_config=meta,
                inv_response_dict=inv_intent_dict,
                encoded_all_responses=encoded_all_intents,
                session=sess,
                graph=graph,
                message_placeholder=a_in,
                response_placeholder=b_in,
                similarity_op=sim_op,
                word_embed=word_embed,
                response_embed=intent_embed,
            )

        else:
            logger.warning(
                "Failed to load nlu model. Maybe path {} "
                "doesn't exist"
                "".format(os.path.abspath(model_dir))
            )
            return cls(component_config=meta)
