package spellchecker.neural_net;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import util.Helper;
import util.Seq2Seq;

import java.io.IOException;

public class EmbeddingRNN extends Seq2Seq {

    public static Seq2Seq Builder(){
        return new EmbeddingRNN();
    }

    public void runTraining() throws IOException { // TODO, wrong matrix dimensions
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
//                    System.out.println("kabbelåjeløkken --> " + generateSuggestion("kabbelåjeløkken").trim() + "\n(kabbelejeløkken)");
                }
            }
            if(i % 5 == 0) System.out.println("Finished EPOCH #" + i);
            if(i % 10 == 0)  ModelSerializer.writeModel(net, String.format("data/models/%s%s.bin", baseFilename, i), true);
            itr.reset();    //Reset iterator for another epoch
        }
        ModelSerializer.writeModel(net, "embModel.bin", true);
    }

    @Override
    public void runTesting(boolean print) {
        Evaluation eval = new Evaluation(itr.getNbrClasses());
        while(itr.hasNextTest()){
            DataSet ds = itr.nextTest();
            net.rnnClearPreviousState();
            INDArray output = net.output(ds.getFeatures(), false);
            eval.evalTimeSeries(ds.getLabels(), output, ds.getLabelsMaskArray());
            createReadableStatistics(ds.getFeatures(), output, ds.getLabels(), print);
        }
        printStats();
        System.out.println(Helper.reduceEvalStats(eval.stats()));
    }

    @Override
    public Seq2Seq buildNetwork() throws Exception {
        int nClasses = CharacterIterator.getDanishCharacterSet().length;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.RELU)
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(nClasses).nOut(10).build())
                .layer(1, new GravesLSTM.Builder().nIn(10).nOut(15).activation(Activation.SOFTSIGN).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(15).nOut(itr.inputColumns()).activation(Activation.SOFTMAX).build())
                .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
                .inputPreProcessor(1, new FeedForwardToRnnPreProcessor())
                .build();
        net = new MultiLayerNetwork(conf);
        return this;
    }
}
