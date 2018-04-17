package neural_nets.anomalies;

import neural_nets.Seq2Seq;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import symspell.SymSpell;
import util.StringUtils;

public class AnomaliesRNN extends Seq2Seq {

    public static Seq2Seq Builder(){
        return new AnomaliesRNN();
    }

    @Override
    public Seq2Seq buildNetwork() throws Exception {
        int nOut = trainItr.totalOutcomes(), nIn = trainItr.inputColumns();
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .weightInit(WeightInit.RELU)
                .updater(Updater.ADAM)
                .regularization(true)//.l2(1e-1)
                .dropOut(0.2)
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(nIn).nOut(20).build())
                .layer(1, new LSTM.Builder().weightInit(WeightInit.RELU)
                        .nIn(20).nOut(10).activation(Activation.SOFTSIGN).build())
                .layer(2, new LSTM.Builder().weightInit(WeightInit.RELU)
                        .nIn(10).nOut(5).activation(Activation.SOFTSIGN).build())
                .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                        .nIn(5).nOut(nOut).build())
                .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
                .inputPreProcessor(1, new FeedForwardToRnnPreProcessor())
                .build();
        net = new MultiLayerNetwork(config);
        return this;
    }

    public void runTesting(boolean print){
        Evaluation eval = new Evaluation(testItr.totalOutcomes());
        while(testItr.hasNext()){
            DataSet ds = testItr.next();
            net.rnnClearPreviousState();
            INDArray output = net.output(ds.getFeatures(), false, ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
            eval.evalTimeSeries(ds.getLabels(), output, ds.getLabelsMaskArray());
        }
        System.out.println(eval.stats());
    }

    @Override
    public void printSuggestion(String input, SymSpell symSpell) {
        return;
    }
}
