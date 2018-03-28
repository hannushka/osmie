package speed.neural_net;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import util.Helper;
import util.Seq2Seq;
import util.StringUtils;

import java.io.IOException;

public class SpeedNetwork extends Seq2Seq {

    public static int embeddingLayerSize = 20;
    public static int nbrOfSpeedClasses = 3;

    public static Seq2Seq Builder(){
        return new SpeedNetwork();
    }

    @Override
    public Seq2Seq setNbrLayers(int... lstmLayerSize){
        int[] tmp1 = new int[]{embeddingLayerSize};
        int[] tmp2 = lstmLayerSize;
        this.layerDimensions = ArrayUtils.addAll(tmp1, tmp2);
        return this;
    }

    @Override
    public void runTraining() throws IOException {
        int miniBatchNumber = 0, generateSamplesEveryNMinibatches = 100;
        for (int i = 0; i < numEpochs; i++) {
            while (itr.hasNext()) {
                DataSet ds = itr.next();
                net.rnnClearPreviousState();
                net.fit(ds);

                if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
                    System.out.println("--------------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size "
                            + miniBatchSize + " words");
                }
            }
            if(i % 5 == 0) System.out.println("Finished EPOCH #" + i);
            if(i % 10 == 0)  ModelSerializer.writeModel(net, String.format("data/model_speed/model%s.bin", i), true);
            itr.reset();    //Reset iterator for another epoch

        }
        itr.reset();
        runTesting(true);
        ModelSerializer.writeModel(net, "model.bin", true);
    }

    @Override
    public void runTesting(boolean print) {
        Evaluation eval = new Evaluation(itr.getNbrClasses());
        while(itr.hasNextTest()){
            DataSet ds = itr.nextTest();
            net.rnnClearPreviousState();
            INDArray output = net.output(ds.getFeatures(), false, ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
            eval.evalTimeSeries(ds.getLabels(), output, ds.getLabelsMaskArray());
        }
        System.out.println(StringUtils.reduceEvalStats(eval.stats()));
    }

    @Override
    public Seq2Seq buildNetwork() throws Exception{
        int tbpttLength = 50, nOut = itr.totalOutcomes(), nIn = itr.inputColumns();
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(learningRate)
                .seed(12345)
                .regularization(true)
                .l2(1e-6)// Dropout vs l2..!
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)   // TODO swap fro ADAM? RMSPROP?
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(nIn).nOut(embeddingLayerSize).build());

        for(int i = 1; i < layerDimensions.length; i++)
            builder.layer(i, new GravesLSTM.Builder().nIn(layerDimensions[i-1]).nOut(layerDimensions[i]).activation(Activation.TANH).build());  //10->5,5->2

        MultiLayerConfiguration config = builder.layer(layerDimensions.length, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                .nIn(layerDimensions[layerDimensions.length-1]).nOut(nOut).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
                .inputPreProcessor(1, new FeedForwardToRnnPreProcessor())
                .pretrain(false).backprop(true)
                .build();
        net = new MultiLayerNetwork(config);
        return this;

    }
}
