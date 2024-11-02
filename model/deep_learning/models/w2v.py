class Word2Vec(object):
    def __init__(
        self,
        context_len=5,
        min_count=None,
        skip_gram=False,
        max_tokens=None,
        embedding_dim=300,
        filter_stopwords=True,
        noise_dist_power=0.75,
        init="glorot_uniform",
        num_negative_samples=64,
        optimizer="SGD(lr=0.1)",
    ):
        """
        A word2vec model supporting both continuous bag of words (CBOW) and
        skip-gram architectures, with training via noise contrastive
        estimation.

        Parameters:
        -----------
        context_len : int
            The number of words to the left and right of the current word to
            use as context during training. Larger values result in more
            training examples and thus can lead to higher accuracy at the
            expense of additional training time. Default is 5. Known as c in paper.
        min_count : int or None
            Minimum number of times a token must occur in order to be included
            in vocab. If None, include all tokens from `corpus_fp` in vocab.
            Default is None. This will help shrink vocab size ignore low frequency word.
        skip_gram : bool
            Whether to train the skip-gram or CBOW model. The skip-gram model
            is trained to predict the target word i given its surrounding
            context, ``words[i - context:i]`` and ``words[i + 1:i + 1 +
            context]`` as input. Default is False.
        max_tokens : int or None
            Only add the first `max_tokens` most frequent tokens that occur
            more than `min_count` to the vocabulary.  If None, add all tokens
            that occur more than `min_count`. Default is None.
        embedding_dim : int
            The number of dimensions in the final word embeddings. Default is
            300.
        filter_stopwords : bool
            Whether to remove stopwords before encoding the words in the
            corpus. Default is True. Remove meaningless word.
        noise_dist_power : float
            The power the unigram count is raised to when computing the noise
            distribution for negative sampling. A value of 0 corresponds to a
            uniform distribution over tokens, and a value of 1 corresponds to a
            distribution proportional to the token unigram counts. Default is
            0.75.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is 'glorot_uniform'.
        num_negative_samples: int
            The number of negative samples to draw from the noise distribution
            for each positive training sample. If 0, use the hierarchical
            softmax formulation of the model instead. Default is 5.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the `update` method.  If None, use the
            :class:`numpy_ml.neural_nets.optimizers.SGD` optimizer with
            default parameters. Default is None.

        Attributes:
        -----------
        parameters : dict
        hyperparameters : dict
        derived_variables : dict
        gradients : dict

        Notes:
        ------
        The word2vec model is outlined in [1].

        CBOW architecture::

            w_{t-R}   ----|
            w_{t-R+1} ----|
            ...            --> Average --> Embedding layer --> [NCE Layer / HSoftmax] --> P(w_{t} | w_{...})
            w_{t+R-1} ----|
            w_{t+R}   ----|

        Skip-gram architecture::

                                                                   |-->  P(w_{t-R} | w_{t})
                                                                   |-->  P(w_{t-R+1} | w_{t})
            w_{t} --> Embedding layer --> [NCE Layer / HSoftmax] --|     ...
                                                                   |-->  P(w_{t+R-1} | w_{t})
                                                                   |-->  P(w_{t+R} | w_{t})

        where :math:`w_{i}` is the one-hot representation of the word at position
        `i` within a sentence in the corpus and `R` is the length of the context
        window on either side of the target word.
        """

        self.init = init
        self.optimizer = optimizer
        self.skip_gram = skip_gram
        self.min_count = min_count
        self.max_tokens = max_tokens
        self.context_len = context_len
        self.embedding_dim = embedding_dim
        self.filter_stopwords = filter_stopwords
        self.noise_dist_power = noise_dist_power
        self.num_negative_samples = num_negative_samples
        self.special_chars = {"<unk>", "<eol>", "<bol>"}

    def _init_params(self):
        self._dv = {}
        self._build_noise_distribution()

        # initial embedding layer same as input layer
        # ex: vocabulary size v = 1000, embedding_dim = 300, w -> (1000, 300)
        self.embeddings = Embedding(
            init=self.init,
            vocab_size=self.vocab_size,
            n_out=self.embedding_dim,
            optimizer=self.optimizer,
            pool=None if self.skip_gram else "mean",
        )
        # initial cross entropy loss, ex: output predict class -> vocabulary size v = 1000
        self.loss = NCELoss(
            init=self.init,
            optimizer=self.optimizer,
            n_classes=self.vocab_size,
            subtract_log_label_prob=False,
            noise_sampler=self._noise_sampler,
            num_negative_samples=self.num_negative_samples,
        )

    def forward(self, x, targets, retain_derived=True):
        """
        Evaluate the network on a single minibatch.

        Parameters:
        -----------
        x : :py:class:`ndarray <numpy.ndarray>` of shape `(n_samples, n_in)`
            Layer input, representing a minibatch of `n_samples` examples, each
            consisting of `n_in` integer word indices
        targets : :py:class:`ndarray <numpy.ndarray>` of shape `(n_samples,)`
            Target word index for each example in the minibatch.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If `False`, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            True.

        Returns:
        --------
        loss : float
            The loss associated with the current minibatch
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(n_samples,)`
            The conditional probabilities of the words in `targets` given the
            corresponding example / context in `X`.
        """
        # ex: x -> (8, 1000) here 8 means 8 samples each sample will contains context
        x_emb = self.embeddings.forward(x, retain_derived=True)
        loss, y_pred = self.loss.loss(x_emb, targets.flatten(), retain_derived=True)
        return loss, y_pred

    @property
    def parameters(self):
        """
        Model parameters
        """
        param = {"components": {"embeddings": {}, "loss": {}}}
        if hasattr(self, "embeddings"):
            param["components"] = {
                "embeddings": self.embeddings.parameters,
                "loss": self.loss.parameters,
            }
        return param

    @property
    def hyperparameters(self):
        """
        Model hyperparameters
        """
        hp = {
            "layer": "Word2Vec",
            "init": self.init,
            "skip_gram": self.skip_gram,
            "optimizer": self.optimizer,
            "max_tokens": self.max_tokens,
            "context_len": self.context_len,
            "embedding_dim": self.embedding_dim,
            "noise_dist_power": self.noise_dist_power,
            "filter_stopwords": self.filter_stopwords,
            "num_negative_samples": self.num_negative_samples,
            "vocab_size": self.vocab_size if hasattr(self, "vocab_size") else None,
            "components": {"embeddings": {}, "loss": {}},
        }

        if hasattr(self, "embeddings"):
            hp["components"] = {
                "embeddings": self.embeddings.hyperparameters,
                "loss": self.loss.hyperparameters,
            }
        return hp

    @property
    def derived_variables(self):
        """
        Variables computed during model operation
        """
        dv = {"components": {"embeddings": {}, "loss": {}}}
        dv.update(self._dv)

        if hasattr(self, "embeddings"):
            dv["components"] = {
                "embeddings": self.embeddings.derived_variables,
                "loss": self.loss.derived_variables,
            }
        return dv

    @property
    def gradients(self):
        """
        Model parameter gradients
        """
        grad = {"components": {"embeddings": {}, "loss": {}}}
        if hasattr(self, "embeddings"):
            grad["components"] = {
                "embeddings": self.embeddings.gradients,
                "loss": self.loss.gradients,
            }
        return grad