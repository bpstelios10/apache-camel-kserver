package org.learnings.aimodels.sentiments.web.api;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.util.ClassLoaderUtils;
import com.google.protobuf.ByteString;
import inference.GrpcPredictV2.InferTensorContents;
import inference.GrpcPredictV2.ModelInferRequest;
import inference.GrpcPredictV2.ModelInferResponse;
import org.apache.camel.Exchange;
import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.model.rest.RestBindingMode;
import org.apache.camel.model.rest.RestParamType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URL;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.stream.Collectors;

public class BertBaseUncasedRoute extends RouteBuilder {

    Logger log = LoggerFactory.getLogger(BertBaseUncasedRoute.class);
    private final HuggingFaceTokenizer bertTokenizer = HuggingFaceTokenizer.newInstance("bert-base-uncased");
    private final DefaultVocabulary vocabulary;
    private int maskIndex = -1;

    public BertBaseUncasedRoute() throws IOException {
        URL resource = ClassLoaderUtils.getResource("vocab.txt");
        vocabulary = DefaultVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(resource)
                .optUnknownToken("[UNK]")
                .build();
    }

    @Override
    public void configure() {
        // Configure REST via Undertow
        restConfiguration()
                .component("undertow")
                .host("0.0.0.0")
                .port(8080)
                .bindingMode(RestBindingMode.off);

        // REST GET endpoint
        rest("/next-sentence-prediction")
                .get()
                .param().name("sentence").required(true).type(RestParamType.query).endParam()
                .outType(String[].class)
                .responseMessage().code(200).message("the next sentence is.. ").endResponseMessage()
                .to("direct:classify");

        // Main route
        from("direct:classify")
                .routeId("bert-inference")
                .setBody(this::createRequest)
                .setHeader("Content-Type", constant("application/json"))
                //                .to("kserve:infer?modelName=bert-base-uncased&target=host.docker.internal:8001")
                .to("kserve:infer?modelName=bert-base-uncased&target=localhost:8001")
                .process(this::postProcess);
    }

    private ModelInferRequest createRequest(Exchange exchange) {
        String sentence = exchange.getIn().getHeader("sentence", String.class);
        maskIndex = sentence.indexOf("[MASK]");
        Encoding encoding = bertTokenizer.encode(sentence);
        List<Long> inputIds = Arrays.stream(encoding.getIds()).boxed().toList();
        List<Long> attentionMask = Arrays.stream(encoding.getAttentionMask()).boxed().toList();
        List<Long> tokenTypeIds = Arrays.stream(encoding.getTypeIds()).boxed().toList();
        maskIndex = Arrays.stream(encoding.getTokens()).toList().indexOf("[MASK]");

        var content0 = InferTensorContents.newBuilder().addAllInt64Contents(inputIds);
        var input0 = ModelInferRequest.InferInputTensor.newBuilder()
                .setName("input_ids").setDatatype("INT64").addShape(1).addShape(inputIds.size())
                .setContents(content0);

        var content1 = InferTensorContents.newBuilder().addAllInt64Contents(attentionMask);
        var input1 = ModelInferRequest.InferInputTensor.newBuilder()
                .setName("attention_mask").setDatatype("INT64").addShape(1).addShape(attentionMask.size())
                .setContents(content1);

        var content2 = InferTensorContents.newBuilder().addAllInt64Contents(tokenTypeIds);
        var input2 = ModelInferRequest.InferInputTensor.newBuilder()
                .setName("token_type_ids").setDatatype("INT64").addShape(1).addShape(tokenTypeIds.size())
                .setContents(content2);

        ModelInferRequest requestBody = ModelInferRequest.newBuilder()
                .addInputs(0, input0).addInputs(1, input1).addInputs(2, input2)
                .build();
        log.debug("-- payload: [{}]", requestBody);

        return requestBody;
    }

    private void postProcess(Exchange exchange) {
        log.debug("-- in response");
        ModelInferResponse response = exchange.getMessage().getBody(ModelInferResponse.class);

        List<List<Float>> logits = response.getRawOutputContentsList().stream()
                .map(ByteString::asReadOnlyByteBuffer)
                .map(buf -> buf.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer())
                .map(buf -> {
                    List<Float> longs = new ArrayList<>(buf.remaining());
                    while (buf.hasRemaining()) {
                        longs.add(buf.get());
                    }
                    return longs;
                })
                .toList();

        log.debug("-- logits: [{}]", logits);

        int[] topIndices = topKFromFlatLogits(logits.getFirst(), maskIndex, (int) vocabulary.size(), 5);

        String result = "Top 5 predictions: " +
                        Arrays.stream(topIndices)
                                .mapToObj(vocabulary::getToken)
                                .collect(Collectors.joining(", "));

        exchange.getMessage().setBody(result);
    }

    public static int[] topKFromFlatLogits(List<Float> flatLogits, int maskIndex, int vocabSize, int k) {
        int offset = maskIndex * vocabSize;
        PriorityQueue<int[]> minHeap = new PriorityQueue<>(Comparator.comparingDouble(a -> a[1]));

        for (int i = 0; i < vocabSize; i++) {
            float logit = flatLogits.get(offset + i);
            if (minHeap.size() < k) {
                minHeap.offer(new int[]{i, Float.floatToIntBits(logit)});
            } else if (Float.intBitsToFloat(minHeap.peek()[1]) < logit) {
                minHeap.poll();
                minHeap.offer(new int[]{i, Float.floatToIntBits(logit)});
            }
        }

        // Extract and sort descending
        int[] result = new int[k];
        for (int i = k - 1; i >= 0; i--) {
            result[i] = minHeap.poll()[0];
        }

        return result;
    }
}
