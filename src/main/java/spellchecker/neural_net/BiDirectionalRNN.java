package spellchecker.neural_net;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import util.Seq2Seq;

public class BiDirectionalRNN extends RNN {

    public static Seq2Seq Builder(){
        return new BiDirectionalRNN();
    }

    @Override
    public Seq2Seq buildNetwork() throws Exception {
        int nOut = itr.totalOutcomes(), idx = 1;
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(learningRate)
                .seed(12345)
                .regularization(true).l2(1e-4)// Dropout vs l2..!
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)   // TODO swap fro ADAM? RMSPROP?
                .list()
                .layer(0, new GravesBidirectionalLSTM.Builder().nIn(nOut).nOut(layerDimensions[0]).activation(Activation.SOFTSIGN).build());//
        for(int i = 1; i < layerDimensions.length; i++, idx++)
            builder.layer(idx, new GravesBidirectionalLSTM.Builder().nIn(layerDimensions[i-1]).nOut(layerDimensions[i]).activation(Activation.SOFTSIGN).build());  //10->5,5->2

        for(int i = layerDimensions.length - 1; i > 0; i--, idx++)
            builder.layer(idx, new GravesBidirectionalLSTM.Builder().nIn(layerDimensions[i]).nOut(layerDimensions[i-1]).activation(Activation.SOFTSIGN).build());

        MultiLayerConfiguration config =  builder.layer(idx, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                .nIn(layerDimensions[0]).nOut(nOut).build())
                .build();
        net = new MultiLayerNetwork(config);
        return this;
    }

}
