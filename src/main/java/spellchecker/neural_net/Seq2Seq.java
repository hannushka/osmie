package spellchecker.neural_net;

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

import java.io.IOException;
import java.util.Collections;

public abstract class Seq2Seq {
    public enum ScoreListener{
        VISUALIZE,
        TERMINAL
    }
    public enum IteratorType{
        TRANSLATER,
        EMBEDDING,
        SPEED
    }
    protected int[] lstmLayerSize = new int[]{10};                    //Number of units in each GravesLSTM layer
    protected int miniBatchSize = 32, numEpochs = 50, epochSize = Integer.MAX_VALUE; //Size of mini batch to use when training
    private int nCharactersToSample = 50;
    protected double learningRate = 0.01;
    protected MultiLayerNetwork net;
    protected CharacterIterator itr;
    private int noChangeCorrect = 0, noChangeIncorrect = 0, changedCorrectly = 0, changedIncorrectly = 0;
    private int wrongChangeType = 0;

    public Seq2Seq setNbrLayers(int... lstmLayerSize){
        this.lstmLayerSize = lstmLayerSize;
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
            case TRANSLATER:
                itr = Helper.getCharacterIterator(miniBatchSize, exampleLength, numEpochs, minimized);
                break;
            case EMBEDDING:
                itr = Helper.getEmbeddedIterator(miniBatchSize, exampleLength, numEpochs, minimized);
                break;
            case SPEED:
                itr = Helper.getSpeedIterator(miniBatchSize, exampleLength, numEpochs, minimized);
                break;
        }
        return this;
    }

    public Seq2Seq setScoreListener(ScoreListener type) throws Exception{
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

    public Seq2Seq loadModel(String filepath) throws IOException {
        net = ModelSerializer.restoreMultiLayerNetwork(filepath, true);
        return this;
    }

    public String generateSuggestion(String street){
        INDArray initializationInput = Nd4j.zeros(1, itr.inputColumns(), street.length());
//        INDArray inputMask = Nd4j.zeros(numSamples, initialization.length());

        char[] init = street.toCharArray();
        Collections.reverse(Chars.asList(init));
        for(int i = 0; i < init.length; i++){
            int idx = itr.convertCharacterToIndex(init[i]);
            initializationInput.putScalar(new int[]{0,idx,i}, 1.0f);
        }

        StringBuilder sb = new StringBuilder(street);

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState();
//            INDArray output = net.output(initializationInput, false);
//            System.out.println(convertTensorsToWords(output)[0]);

        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2)-1,1,0);
        //Gets the last time step output

        for(int i = 0; i < nCharactersToSample; i++){
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(1, itr.inputColumns());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            double[] outputProbDistribution = new double[itr.totalOutcomes()];
            for(int j = 0; j < outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(0,j);
            int sampledCharacterIdx = Helper.getMax(outputProbDistribution);

            nextInput.putScalar(new int[]{0,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
            if(sampledCharacterIdx == -1) sb.append("<NaN>");
            else sb.append(itr.convertIndexToCharacter(sampledCharacterIdx));

            output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
        }
        return sb.toString();
    }

    void createReadableStatistics(INDArray input, INDArray result, INDArray labels, boolean print){
        // No change correct
        // No change incorrect
        // Change correct
        // Change correct but changed wrong
        // Change incorrect
        if(!print) return;
        String[] inputStr = Helper.convertTensorsToWords(input, itr, nCharactersToSample);
        String[] resultStr = Helper.convertTensorsToWords(result, itr, nCharactersToSample);
        String[] labelStr = Helper.convertTensorsToWords(labels, itr, nCharactersToSample);
        String inp, out, label;
        for(int i = 0; i < inputStr.length; i++){
            inp = inputStr[i].replaceAll("<NaN>", "").trim();
            out = resultStr[i].replaceAll("<NaN>", "").trim();
            label = labelStr[i].replaceAll("<NaN>", "").trim();
            System.out.println(inp + ",,," + out + ",,," + label);
            if(inp.equals(out) && inp.equals(label)) noChangeCorrect++;
            if(inp.equals(out) && !inp.equals(label)) noChangeIncorrect++;
            if(!inp.equals(label) && out.equals(label)) changedCorrectly++;
            if(inp.equals(label) && !inp.equals(out)) changedIncorrectly++;
            if(!inp.equals(label) && !inp.equals(out) && !out.equals(label))  wrongChangeType++;
        }
    }

    void printStats(){
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

    public abstract void runTraining() throws IOException;

    public abstract void runTesting(boolean print);

    public abstract Seq2Seq buildNetwork() throws Exception;

//    public void runTestingOnTrain(boolean print){
//        itr.reset();
//        Evaluation eval = new Evaluation(itr.getNbrClasses());
//        while(itr.hasNext()){
//            DataSet ds = itr.next();
//            INDArray output = net.output(ds.getFeatures(), false, ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
//            eval.evalTimeSeries(ds.getLabels(), output, ds.getLabelsMaskArray());
//            createReadableStatistics(ds.getFeatures(), output, ds.getLabels(), print);
//        }
//        printStats();
//        //System.out.println(eval.stats(false));
//    }
}
