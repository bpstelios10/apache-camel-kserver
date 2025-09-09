# Service

The service is an example of a Java app interacting with an ONNX model, using Apache KServe Component.

1) The Java app is using REST via apache-camel Undertow
2) the ONNX model is hosted on a Triton NVIDIA Server (locally)
3) the latest apache-camel KServe component is used for the integration

More detail about KServe can be also found on my article
about [Java apache-camel-kserve component](https://www.baeldung.com/java-apache-camel-kserve).

## Triton Server

The [Triton NVIDIA Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_1-model_deployment/README.html)
was selected because it is a host for pre-trained models, that:

- accepts [ONNX models](https://onnx.ai/)
- supports the
  standard [V2 Inference Protocol](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol)
- offers scalability easily, with deployment in Kubernetes native frameworks,
  like [KServe (KFServing)](https://www.kubeflow.org/docs/components/kserve/introduction/)

## The models

Both models are pre-trained and retrieved from hugging-face. They are used only for learning purposes here.
Both models need to be downloaded from hugging-face and put into the right folder to work.
The models are available through triton, by running _docker-compose_.

### Bert - Mask

I ve used [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main) model that finds the
possible words to replace the "[MASK]" in a sentence. For example, we pass "Have a nice [MASK]" and in the top outputs
we expect to see "have a nice day".

First downloaded the onnx file and put it
in [triton-server/models/bert-base-uncased/1](triton-server/models/bert-base-uncased/1). Then downloaded vocab.txt and
put it into [sentiment-service/src/main/resources](sentiment-service/src/main/resources). Finally, the tokenizer.json is
being used by the line `HuggingFaceTokenizer.newInstance("bert-base-uncased")` (no action needed).

**Model Input using Java**
For the input, i've used the `HuggingFaceTokenizer` class, provided by ai.djl and huggingface. There's an implementation
for Bert that makes our life easy! This will convert the words to ids that the model can understand.

**Model Output using Java**
The output is trickier. The server response comes back as an array of floats. Each value represents the probability of
each word (from the vocabulary). As we said the response is the whole sentence with the "[MASK]" replaced by each
possible word. Then a probability is assigned. This means, if we pass in the sentence "have a nice [MASK]", and the
model vocabulary has 10 words, then the response is the sentence 10 times, one for each word. the word has 6 tokens (i
think), for the 4 words and a start and an end (?). so the response would be 60 floats with the probabilities of each
word.

In this service, we read only the values of possible masks, sort the top 5 possible answers and return the 5 words only.

### Sentiments

I ve used [pjxcharya/onnx-sentiment-model](https://huggingface.co/pjxcharya/onnx-sentiment-model/tree/main) model that
accepts a sentence and returns 2 values, the possibility of this sentence to be good and bad.

First downloaded the onnx file and put it in [triton-server/models/sentiment/1](triton-server/models/sentiment/1).
Finally, there is no tokenizer.json but we can use the same from before here:
`HuggingFaceTokenizer.newInstance("bert-base-uncased")` (no action needed).

**Model Input using Java**
For the input, i've used the `HuggingFaceTokenizer` class, provided by ai.djl and huggingface. I'm using the
implementation for Bert that works here too. This will convert the words to ids that the model can understand.

**Model Output using Java**
As explained, the response is only 2 floats, one for the possibility the sentence to have a bad sentiment and one to
have a good one.

In this service, we compare the 2 values and just return back the outcome, good or bad.

## Docker Compose

Docker-compose is provided to easily start locally the triton server with the pre-trained models loaded. Start the
Triton server only running `docker-compose up triton-server --build`.

## Postman

A Postman collection is provided, that includes REST calls for both the Triton Server directly, but also for our
service.
