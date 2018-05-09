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
import util.Helper;
import util.StringUtils;

import java.util.Arrays;
import java.util.stream.Collectors;

import static util.Helper.getDoubleMatrixDistr;

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

    public double runTesting(boolean print){
        Evaluation eval = new Evaluation(testItr.totalOutcomes());
        while(testItr.hasNext()){
            DataSet ds = testItr.next();
            net.rnnClearPreviousState();
            INDArray output = net.output(ds.getFeatures(), false, ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
            eval.evalTimeSeries(ds.getLabels(), output, ds.getLabelsMaskArray());
            for(int i = 0; i < output.shape()[0]; i++) {
                double[][] wordMatrix = Helper.getDoubleMatrixDistr(output.tensorAlongDimension(i, 1, 2));
                double[] tmp = wordMatrix[wordMatrix.length-1];
                int classRes = Helper.getIndexOfMax(tmp);
                if (classRes == 0) {
                    System.out.println(((AnomaliesIterator)testItr).getLastBatchContainers().get(i).toString());
                }
            }
        }
        System.out.println(eval.stats());
        return eval.f1();
    }

    @Override
    public void printSuggestion(String input, SymSpell symSpell) {
        return;
    }
}
