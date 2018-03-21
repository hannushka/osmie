package speed.neural_net;

import com.google.common.primitives.Chars;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.nio.charset.Charset;

public class SpeedNetwork {
    public enum ScoreListener{
        VISUALIZE,
        TERMINAL
    }
    public enum IteratorType{
        TRANSLATER,
        AUTOENCODE
    }
    private int lstmLayerSize = 10;                    //Number of units in each GravesLSTM layer
    private int miniBatchSize = 32;                        //Size of mini batch to use when training
    private int numEpochs = 50;                            //Total number of training epochs
    private int nCharactersToSample = 50;
    private double learningRate = 0.1;
    private MultiLayerNetwork net;
    private ChSpeedIterator itr;
    private int noChangeCorrect = 0, noChangeIncorrect = 0, changedCorrectly = 0, changedIncorrectly = 0;
    private int wrongChangeType = 0;


    public static SpeedNetwork Builder(){
        return new SpeedNetwork();
    }

    public SpeedNetwork buildNetwork() throws Exception {
        if(itr == null) throw new Exception("itr == null!");
        MultiLayerConfiguration config = buildNetworkConfig(itr);
        net = new MultiLayerNetwork(config);
        return this;
    }

    public SpeedNetwork buildBiNetwork() throws Exception {
        if(itr == null) throw new Exception("itr == null!");
        MultiLayerConfiguration config = buildBiNetworkConfig(itr);
        net = new MultiLayerNetwork(config);
        return this;
    }

    public SpeedNetwork setNbrLayers(int lstmLayerSize){
        this.lstmLayerSize = lstmLayerSize;
        return this;
    }

    public SpeedNetwork setLearningRate(double learningRate){
        this.learningRate = learningRate;
        return this;
    }

    public SpeedNetwork setBatchSize(int miniBatchSize){
        this.miniBatchSize = miniBatchSize;
        return this;
    }

    public SpeedNetwork setNbrEpochs(int numEpochs){
        this.numEpochs = numEpochs;
        return this;
    }

    public SpeedNetwork setCharacterIterator(IteratorType type, int epochSize, boolean minimized) throws Exception {
        int exampleLength = 50;
        switch (type){
            case TRANSLATER:
                itr = getShakespeareIterator(miniBatchSize, exampleLength, epochSize, minimized);
                break;
            case AUTOENCODE:
                itr = getShakespeareIterator(miniBatchSize, exampleLength, epochSize, minimized);
                break;
        }
        return this;
    }

    public SpeedNetwork setScoreListener(ScoreListener type) throws Exception{
        if(net == null) throw new Exception("net == null");

        switch (type){
            case VISUALIZE:
                UIServer uiServer = UIServer.getInstance();
                StatsStorage statsStorage = new InMemoryStatsStorage(); // Alternative use FileStatsStorage for saving&loading
                uiServer.attach(statsStorage);
                net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(25));
                break;
            case TERMINAL:
                net.setListeners(new ScoreIterationListener(1000));
                break;
        }
        return this;
    }

    public SpeedNetwork loadModel(String filepath) throws IOException {
        net = ModelSerializer.restoreMultiLayerNetwork(filepath, true);
        return this;
    }

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
            if(i % 10 == 0)  ModelSerializer.writeModel(net, String.format("data/models/model%s.bin", i), true);
            itr.reset();    //Reset iterator for another epoch
        }
        ModelSerializer.writeModel(net, "model.bin", true);
    }

    private void printStats(){
        System.out.println("====");
        System.out.println("Correctly unchanged = " + noChangeCorrect);
        System.out.println("Incorrectly unchanged = " + noChangeIncorrect);
        System.out.println("Correctly changed = " + changedCorrectly);
        System.out.println("Incorrectly changed = " + changedIncorrectly);
        System.out.println("Correctly changed but to the wrong class = " + wrongChangeType);
        System.out.println("====");
        System.out.println("Errors introduced: " + changedIncorrectly);
        System.out.println("Corrections introduced: " + changedCorrectly + " (total #correct: "
                + (changedCorrectly+noChangeCorrect) + ")");
        System.out.println("Correction that introduce no new error: " + wrongChangeType);
        System.out.println("Unchanged: " + (noChangeCorrect+noChangeIncorrect));
    }

    public void runTesting(boolean print){
        Evaluation eval = new Evaluation(itr.getNbrClasses());
        while(itr.hasNextTest()){
            DataSet ds = itr.nextTest();
            net.rnnClearPreviousState();
            INDArray output = net.output(ds.getFeatures(), false, ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
            eval.evalTimeSeries(ds.getLabels(), output, ds.getLabelsMaskArray());
        }
        //eval.stats().split("\n")
        printStats();
        System.out.println(eval.stats());
//        System.out.println(eval.confusionToString());
        //System.out.println(eval.f1(EvaluationAveraging.Micro));
    }

    public void runTestingOnTrain(boolean print){
        itr.reset();
        Evaluation eval = new Evaluation(itr.getNbrClasses());
        while(itr.hasNext()){
            DataSet ds = itr.next();
            INDArray output = net.output(ds.getFeatures(), false, ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
            eval.evalTimeSeries(ds.getLabels(), output, ds.getLabelsMaskArray());
        }
        printStats();
        //System.out.println(eval.stats(false));
    }


    private int getMax(double[] distribution){
        int idx = 0;
        double max = 0;
        for(int i = 0; i < distribution.length; i++){
            if(distribution[i] > max){
                max = distribution[i];
                idx = i;
            }
        }
        if(max == 0) return -1;
        return idx;
    }

    private MultiLayerConfiguration buildBiNetworkConfig(DataSetIterator iter){
        int nOut = itr.totalOutcomes();
        return new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(learningRate)
                .seed(12345)
                .regularization(true).l2(1e-4)// Dropout vs l2..!
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)   // TODO swap fro ADAM? RMSPROP?
                .list()//
//                .layer(0, new EmbeddingLayer.Builder().nIn(iter.inputColumns()))
                .layer(0, new GravesBidirectionalLSTM.Builder().nIn(itr.inputColumns()).nOut(10).activation(Activation.SOFTSIGN).build())
                .layer(1, new GravesBidirectionalLSTM.Builder().nIn(10).nOut(lstmLayerSize).activation(Activation.SOFTSIGN).build())   // TODO Add 4th layer. 2 encode, 2 decode
                .layer(2, new GravesBidirectionalLSTM.Builder().nIn(lstmLayerSize).nOut(10).activation(Activation.SOFTSIGN).build())
                .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                        .nIn(10).nOut(nOut).build())
//                .pretrain(false).backprop(true)
                .build();
    }

    private MultiLayerConfiguration buildNetworkConfig(DataSetIterator iter){
        int nOut = itr.totalOutcomes();
        int tbpttLength = 50;
        return new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(learningRate)
                .seed(12345)
                .regularization(true).l2(1e-4)// Dropout vs l2..!
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)   // TODO swap fro ADAM? RMSPROP?
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(10)
                        .activation(Activation.TANH).build())
                .layer(1, new GravesLSTM.Builder().nIn(10).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())   // TODO Add 4th layer. 2 encode, 2 decode
                .layer(2, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(10)
                        .activation(Activation.TANH).build())
                .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                        .nIn(10).nOut(nOut).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .pretrain(false).backprop(true)
                .build();
    }


    private ChSpeedIterator getShakespeareIterator(int miniBatchSize, int sequenceLength, int epochSize,
                                                     boolean minimized) throws Exception {
        String fileLocation = "data/nameDataUnique.csv";
        char[] validCharacters = ChSpeedIterator.getDanishCharacterSet();

        return new ChSpeedIterator(fileLocation, Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, validCharacters, epochSize, minimized);
    }
}
