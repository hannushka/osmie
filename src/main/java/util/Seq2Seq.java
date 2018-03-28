package util;

import com.google.common.primitives.Chars;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import spellchecker.neural_net.CharacterIterator;

import java.io.IOException;
import java.util.Collections;

public abstract class Seq2Seq {
    public enum ScoreListener{
        VISUALIZE,
        TERMINAL,
        ALL
    }
    public enum IteratorType{
        CLASSIC,
        SPEED
    }
    protected int[] layerDimensions = new int[]{}; //Number of units in each GravesLSTM layer
    protected int miniBatchSize = 32, numEpochs = 50, epochSize = Integer.MAX_VALUE; //Size of mini batch to use when training
    private int nCharactersToSample = 50;
    protected double learningRate = 0.01;
    protected String baseFilename = "models";
    protected MultiLayerNetwork net;
    protected CharacterIterator itr;
    private int noChangeCorrect = 0, noChangeIncorrect = 0, changedCorrectly = 0, changedIncorrectly = 0, editDistOne = 0;
    private int wrongChangeType = 0;
    private boolean useCorpus = true;

    public Seq2Seq useCorpus(boolean useCorpus){
        this.useCorpus = useCorpus;
        return this;
    }

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

    public Seq2Seq setCharacterIterator(IteratorType type, boolean minimized) throws Exception {
        int exampleLength = 50;
        switch (type){
            case CLASSIC:
                itr = CharacterIterator.getCharacterIterator(miniBatchSize, exampleLength, epochSize, minimized, useCorpus);
                break;
            case SPEED:
                itr = CharacterIterator.getSpeedIterator(miniBatchSize, exampleLength, epochSize, minimized);
                break;
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
        String[] inputStr = Helper.convertTensorsToWords(input, itr);
        String[] resultStr = Helper.convertTensorsToWords(result, itr);
        String[] labelStr = Helper.convertTensorsToWords(labels, itr);
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
    }

    public abstract void runTraining() throws IOException;

    public abstract void runTesting(boolean print);

    public abstract Seq2Seq buildNetwork() throws Exception;
}
