package neural_nets.anomalies;

import neural_nets.Seq2Seq;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
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
                .regularization(true)
                .l2(1e-5)
                .learningRate(learningRate)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(nIn).nOut(10).build())
                .layer(1, new GravesBidirectionalLSTM.Builder().nIn(10).nOut(5).activation(Activation.SOFTSIGN).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(5).nOut(nOut).build())
                .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
                .inputPreProcessor(1, new FeedForwardToRnnPreProcessor())
                .build();
        net = new MultiLayerNetwork(config);
        return this;
    }

    public void runTesting(boolean print){
        Evaluation eval = new Evaluation(testItr.getNbrClasses());
        while(testItr.hasNext()){
            DataSet ds = testItr.next();
            net.rnnClearPreviousState();
            INDArray output = net.output(ds.getFeatures(), false, ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
            eval.evalTimeSeries(ds.getLabels(), output, ds.getLabelsMaskArray());
            createReadableStatistics(ds.getFeatures(), output, ds.getLabels(), print);
        }
        printStats();
        System.out.println(StringUtils.reduceEvalStats(eval.stats()));
    }
}