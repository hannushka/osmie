package speed.neural_net;

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
import spellchecker.neural_net.Seq2Seq;

import java.io.IOException;

public class SpeedNetwork extends Seq2Seq {

    public static Seq2Seq Builder(){
        return new SpeedNetwork();
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
//                    System.out.println("kabbelåjeløkken --> " + generateSuggestion("kabbelåjeløkken").trim() + "\n(kabbelejeløkken)");
                }
            }
            if(i % 5 == 0) System.out.println("Finished EPOCH #" + i);
            if(i % 10 == 0)  ModelSerializer.writeModel(net, String.format("data/model_speed/model%s.bin", i), true);
            itr.reset();    //Reset iterator for another epoch
        }
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
        //eval.stats().split("\n")
        System.out.println(eval.stats());
    }

    @Override
    public Seq2Seq buildNetwork() throws Exception{
        int tbpttLength = 50, idx = 2, nOut = itr.totalOutcomes(), nIn = itr.inputColumns();
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(learningRate)
                .seed(12345)
                .regularization(true).l2(1e-4)// Dropout vs l2..!
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)   // TODO swap fro ADAM? RMSPROP?
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(nIn).nOut(20).build())
                .layer(1, new GravesLSTM.Builder().nIn(20).nOut(lstmLayerSize[0])
                        .activation(Activation.SOFTSIGN).build());

        for(int i = 1; i < lstmLayerSize.length; i++, idx++)
            builder.layer(idx, new GravesLSTM.Builder().nIn(lstmLayerSize[i-1]).nOut(lstmLayerSize[i]).activation(Activation.SOFTSIGN).build());  //10->5,5->2

        MultiLayerConfiguration config = builder.layer(idx, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                .nIn(lstmLayerSize[0]).nOut(nOut).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
                .inputPreProcessor(1, new FeedForwardToRnnPreProcessor())
                .pretrain(false).backprop(true)
                .build();
        net = new MultiLayerNetwork(config);
        return this;

    }
}
