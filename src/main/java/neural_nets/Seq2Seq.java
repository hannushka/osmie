package neural_nets;

import neural_nets.anomalies.AnomaliesPredictIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import symspell.SymSpell;
import symspell.SuggestItem;
import neural_nets.anomalies.AnomaliesIterator;
import neural_nets.spellchecker.SpellCheckIterator;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import util.DeepSpellObject;
import util.Helper;
import util.StringUtils;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.nio.charset.Charset;

public abstract class Seq2Seq {
    public enum ScoreListener{
        VISUALIZE,
        TERMINAL,
        ALL
    }
    public enum IteratorType{
        CLASSIC,
        ANOMALIES,
        ANOMALIES_PREDICT
    }

    protected int[] layerDimensions = new int[]{}; //Number of units in each GravesLSTM layer
    protected int miniBatchSize = 32, numEpochs = 50, epochSize = Integer.MAX_VALUE; //Size of mini batch to use when training
    protected double learningRate = 0.01;
    protected String baseFilename = "models";
    protected MultiLayerNetwork net;
    protected CharacterIterator trainItr, testItr;
    protected int sCorr = 0, ssCorr = 0, sInc = 0, ssInc = 0;
    protected int noChangeCorrect = 0, noChangeIncorrect = 0, changedCorrectly = 0, changedIncorrectly = 0,
            editDistOne = 0, wrongChangeType = 0;

    public Seq2Seq setFilename(String name){
        this.baseFilename = name;
        return this;
    }

    public Seq2Seq setNbrLayers(int... layerDimensions){
        this.layerDimensions = layerDimensions;
        return this;
    }

    public Seq2Seq setLearningRate(double learningRate){
        this.learningRate = learningRate;
        return this;
    }

    public Seq2Seq setBatchSize(int miniBatchSize){
        this.miniBatchSize = miniBatchSize;
        return this;
    }

    public Seq2Seq setNbrEpochs(int numEpochs){
        this.numEpochs = numEpochs;
        return this;
    }

    public Seq2Seq setEpochSize(int epochSize){
        this.epochSize = epochSize;
        return this;
    }

    public Seq2Seq setCharacterIterator(String fileLocation, String testFileLocation,
                                        IteratorType type, boolean merge, boolean offset) throws Exception {
        int exampleLength = 50;
        switch (type){
            case CLASSIC:
                trainItr = new SpellCheckIterator(fileLocation, Charset.forName("UTF-8"),
                        miniBatchSize, exampleLength, epochSize, merge, offset);
                testItr = new SpellCheckIterator(testFileLocation, Charset.forName("UTF-8"),
                        miniBatchSize, exampleLength, epochSize, merge, offset);
                break;
            case ANOMALIES:
                exampleLength = 5;
                trainItr = new AnomaliesIterator(fileLocation, Charset.forName("UTF-8"),
                        miniBatchSize, exampleLength, epochSize);
                testItr = new AnomaliesIterator(testFileLocation, Charset.forName("UTF-8"),
                        miniBatchSize, exampleLength, epochSize);
                break;
            case ANOMALIES_PREDICT:
                exampleLength = 5;
                testItr = new AnomaliesPredictIterator(testFileLocation, Charset.forName("UTF-8"),
                        miniBatchSize, exampleLength, epochSize);
        }
        return this;
    }

    public Seq2Seq setScoreListener(ScoreListener type) throws Exception{
        if(net == null) throw new Exception("net == null");

        switch (type){
            case VISUALIZE:
                UIServer uiServer = UIServer.getInstance();
                StatsStorage statsStorage = new InMemoryStatsStorage();
                uiServer.attach(statsStorage);
                net.setListeners(new StatsListener(statsStorage));
                break;
            case TERMINAL:
                net.setListeners(new ScoreIterationListener(1000));
                break;
            case ALL:
                uiServer = UIServer.getInstance();
                statsStorage = new InMemoryStatsStorage();
                uiServer.attach(statsStorage);
                net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(25));
                break;
        }
        return this;
    }

    public Seq2Seq loadModel(String filepath) throws IOException {
        net = ModelSerializer.restoreMultiLayerNetwork(filepath, true);
        return this;
    }

    public void createReadableStatistics(INDArray input, INDArray result, INDArray labels, boolean print){
        String[] inputStr = Helper.convertTensorsToWords(input, trainItr);
        String[] resultStr = Helper.convertTensorsToWords(result, trainItr);
        String[] labelStr = Helper.convertTensorsToWords(labels, trainItr);
        String inp, out, label;
        for(int i = 0; i < inputStr.length; i++){
            inp = inputStr[i];
            out = resultStr[i];
            label = labelStr[i];
            if(print) System.out.println(inp + ",,," + out + ",,," + label);
            if(StringUtils.oneEditDist(inp, label)) editDistOne++;
            if(inp.equals(out) && inp.equals(label)) noChangeCorrect++;
            if(inp.equals(out) && !inp.equals(label)) noChangeIncorrect++;
            if(!inp.equals(label) && out.equals(label)) changedCorrectly++;
            if(inp.equals(label) && !inp.equals(out)) changedIncorrectly++;
            if(!inp.equals(label) && !inp.equals(out) && !out.equals(label))  wrongChangeType++;
        }
    }

    protected void printStats(){
        System.out.println("====");
        System.out.println("Errors (good->bad): \t\t" + changedIncorrectly);
        System.out.println("Corrections (bad->good):\t" + changedCorrectly + " (total #correct: "
                + (changedCorrectly+noChangeCorrect) + ")");
        System.out.println("Correction (bad->bad): \t\t" + wrongChangeType);
        System.out.println("Total changes: \t\t\t\t" + (wrongChangeType + changedCorrectly + changedIncorrectly));
        System.out.println("----");
        System.out.println("Unchanged correctly (good->good): \t" + noChangeCorrect);
        System.out.println("Unchanged incorrectly (bad->bad): \t" + noChangeIncorrect);
        System.out.println("Total unchanged: \t\t\t" + (noChangeCorrect+noChangeIncorrect));
        System.out.println("----");
        System.out.println("Edits within one: \t\t\t" + editDistOne);
        System.out.println("----");
        System.out.println(ssCorr + "\t" + ssInc);
        System.out.println(sCorr + "\t" + sInc);
    }

    public void runTraining() throws IOException {
        int miniBatchNumber = 0, generateSamplesEveryNMinibatches = 100;
        for (int i = 0; i < numEpochs; i++) {
            while (trainItr.hasNext()) {
                DataSet ds = trainItr.next();
                net.rnnClearPreviousState();
                net.fit(ds);

                if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
                    System.out.println("--------------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size "
                            + miniBatchSize + " words");
                }
            }
            if(i % 5 == 0) System.out.println("Finished EPOCH #" + i);
            if(i % 10 == 0)  ModelSerializer.writeModel(net, String.format("data/models/%s%s.bin", baseFilename, i), true);
            trainItr.reset();
        }
        ModelSerializer.writeModel(net, "model.bin", true);
    }

    public abstract double runTesting(boolean print);

    public abstract void printSuggestion(String input, SymSpell symSpell);

    public abstract Seq2Seq buildNetwork() throws Exception;

    public abstract void predict() throws Exception;
}
