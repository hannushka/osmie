package speed.neural_net;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import spellchecker.neural_net.CharacterIterator;
import util.Helper;
import util.NgramBuilder;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/** A simple DataSetIterator for use in the GravesLSTMCharModellingExample.
 * Given a text file and a few options, generate feature vectors and labels for training,
 * where we want to predict the next character in the sequence.<br>
 * This is done by randomly choosing a position in the text file, at offsets of 0, exampleLength, 2*exampleLength, etc
 * to start each sequence. Then we convert each character to an index, i.e., a one-hot vector.
 * Then the character 'a' becomes [1,0,0,0,...], 'b' becomes [0,1,0,0,...], etc
 *
 * Feature vectors and labels are both one-hot vectors of same length
 * @author Alex Black
 */
public class ChSpeedIterator extends CharacterIterator {
    private Map<String,Integer> bigramToIdx;
    private int exampleLength, miniBatchSize, numExamples, pointer = 0, epochSize, currEx = 0;
    private ArrayList<ArrayList<String>> inputList, testInputList;
    ArrayList<Integer> outputList, testOutputList;
    NgramBuilder ngramBuilder;

    public ChSpeedIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                            String alphFilePath , int epochSize, boolean minimized) throws IOException {
        if(!new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
        if(miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be > 0)");
        this.exampleLength  = exampleLength;
        this.miniBatchSize  = miniBatchSize;
        this.epochSize      = epochSize;
        this.inputList      = new ArrayList<>();
        this.outputList     = new ArrayList<>();
        this.testInputList  = new ArrayList<>();
        this.testOutputList = new ArrayList<>();
        this.bigramToIdx    = new HashMap<>();
        int j = 0;
        ngramBuilder = new NgramBuilder(2);
        ngramBuilder.createNgramAlphabet("data/nameDataUnique.csv",
                "data/bigramAlphabet.csv", 5);
        ngramBuilder.loadNgramMap("data/bigramAlphabet.csv");
//        List<String> alpha = Files.readAllLines(new File(alphFilePath).toPath(), textFileEncoding);
//        StringJoiner joiner = new StringJoiner("\n");
//        for (String s : alpha) joiner.add(s);
        for (String s : ngramBuilder.nGrams) bigramToIdx.put(s, j++);
        bigramToIdx.put("!!", j);
        System.out.println(bigramToIdx.size());

        j = 0;
        List<String> lines = Files.readAllLines(new File(textFilePath).toPath(), textFileEncoding);
        int splitThreshold = (int) (lines.size() * 0.95);
        if (minimized) splitThreshold = 950;

        for (String s : lines) {
            if (s.isEmpty() || (minimized && j > 1000)) continue;
            j++;
            String[] inputOutput = s.split(",,,");
            if (inputOutput.length < 3) continue;
//            ArrayList<String> bigrams = Helper.getNgrams(inputOutput[1].toLowerCase(), bigramToIdx, 2);
            ArrayList<String> bigrams = ngramBuilder.getFilteredNgrams(inputOutput[1].toLowerCase());
            Integer speed = tryParse(inputOutput[2]);
            if (speed == null) continue;

            if (j > splitThreshold) {
                inputList.add(bigrams);
                outputList.add(speed);
            } else {
                testInputList.add(bigrams);
                testOutputList.add(speed);
            }
        }
        balanceLists();
        numExamples = inputList.size();
    }

    private void balanceLists(){
        Map<Integer, ArrayList<Integer>> speedToIdx = new HashMap<>();
        ArrayList<ArrayList<String>> balancedInput = new ArrayList<>();
        ArrayList<Integer> balancedOutput = new ArrayList<>();
        int minSize = Integer.MAX_VALUE;
        for (int i = 0; i < outputList.size(); i++) {
            ArrayList<Integer> indices = speedToIdx.getOrDefault(outputList.get(i), new ArrayList<>());
            indices.add(i);
            speedToIdx.put(i, indices);
        }
        Set<Integer> keys = new HashSet<>();
        for (Map.Entry<Integer, ArrayList<Integer>> e : speedToIdx.entrySet()){
            if (e.getValue().size() < minSize) minSize = e.getValue().size();
            keys.add(e.getKey());
        }
        for (int i = 0; i < minSize; i ++) {
            for (int key : keys) {
                balancedInput.add(inputList.get(speedToIdx.get(key).get(i)));
                balancedOutput.add(outputList.get(speedToIdx.get(key).get(i)));
            }
        }
        inputList = balancedInput;
        outputList = balancedOutput;
    }

    private static Integer tryParse(String text) {
        try {
            return Integer.parseInt(text);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private Integer intToIdx(int speed){
        switch(speed) {
            case 30: return 0;
            case 70: return 1;
            case 130: return 2;
            default:
                throw new NumberFormatException("Speed " + speed + " not in speed alphabet.");
        }
    }

    /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
    public static char[] getMinimalCharacterSet(){
        List<Character> validChars = new LinkedList<>();
        for(char c='a'; c<='z'; c++) validChars.add(c);
        for(char c='A'; c<='Z'; c++) validChars.add(c);
        for(char c='0'; c<='9'; c++) validChars.add(c);
        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
        for( char c : temp ) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i=0;
        for( Character c : validChars ) out[i++] = c;
        return out;
    }

    public static char[] getDanishCharacterSet(){
        try (BufferedReader br = Files.newBufferedReader(Paths.get("data/dk_alphabet.txt"))) {
            Set<Character> validChars = new HashSet<>();    // TODO Testing lower case..! Easier to start with!
            char[] temp = {'!', '&', '-', '\'', '"', ',', '.', ' ', '\n', '\t', 'ü', 'ë', 'é'};

            for(char c : br.readLine().toLowerCase().toCharArray()) validChars.add(c);
            for(char c : temp) validChars.add(c);       // ^ Adding these here as they are not common to misspell
            char[] out = new char[validChars.size()];
            int i = 0;
            for(char c: validChars) out[i++] = c;

            return out;
        }catch (IOException ex){
            ex.printStackTrace();
        }
        return getMinimalCharacterSet();
    }

    public int getNbrClasses(){
        return SpeedNetwork.nbrOfSpeedClasses;
    }

    public boolean hasNext() {
        return currEx < inputList.size() && pointer < epochSize;
    }

    public boolean hasNextTest() {
        return currEx < testInputList.size();
    }

    public DataSet next() {
        return createDataSet(miniBatchSize, inputList, outputList);
    }

    public DataSet next(int num) {
        return createDataSet(num, inputList, outputList);
    }

    public DataSet createDataSet(int num, ArrayList<ArrayList<String>> in, ArrayList<Integer> out){
        if (in.isEmpty()) throw new NoSuchElementException();
        int currMinibatchSize = Math.min(num, in.size() - currEx);
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        // 'f' (fortran) ordering = must for optimized custom iterator.
        INDArray input = Nd4j.create(new int[]{currMinibatchSize, 1, exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize, SpeedNetwork.nbrOfSpeedClasses, exampleLength}, 'f');
        INDArray inputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
        INDArray outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');

        for (int i = 0; i < currMinibatchSize; i++) {  // Iterating each line
            ArrayList<String> inputNgrams = in.get(currEx);
            int outputToIdx = out.get(currEx);
//            Integer outputToIdx = output;
            currEx++;
            pointer++;

            if (inputNgrams == null) continue;
            outputMask.putScalar(new int[]{i, exampleLength - 1}, 1f);
            for (int j = 0; j < inputNgrams.size() + 1; j++)
                inputMask.putScalar(new int[]{i, j}, 1f);
            labels.putScalar(new int[]{i, outputToIdx, exampleLength - 1}, 1f);

            for (int j = 0; j < exampleLength; j++) {
                int currNgramIdx = bigramToIdx.get("!!");
                if (inputNgrams.size() > j) currNgramIdx = bigramToIdx.getOrDefault(inputNgrams.get(j), currNgramIdx);
                input.putScalar(new int[]{i, 0, j}, currNgramIdx);
            }
        }
        return new DataSet(input, labels, inputMask, outputMask);
    }

    public DataSet nextTest() {
        return createDataSet(miniBatchSize, testInputList, testOutputList);
    }

    public int totalExamples() {
        return numExamples;
    }

    public int inputColumns() {
        return bigramToIdx.size();
    }

    public int totalOutcomes() {
        return SpeedNetwork.nbrOfSpeedClasses;
    }

    public void reset() {
        if(currEx < inputList.size()){
            pointer = 0;
            return;
        }
        currEx = 0;
        pointer = 0;
    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    public int batch() {
        return miniBatchSize;
    }

    public int cursor() {
        return totalExamples() - inputList.size();
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public List<String> getLabels() {
        return null;    // Returning the same as another example from DL4j that uses seq-2-seq
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

}
