package speed.neural_net;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

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
public class ChSpeedIterator implements DataSetIterator {
    private char[] validCharacters;
    //Maps each character to an index in the input/output
    private Map<Character,Integer> charToIdxMap;
    //All characters of the input file (after filtering to only those that are valid)
    private ArrayList<char[]> inputLines, ogInput, inputTest;
    private ArrayList<Integer> outputLines, ogOutput, outputTest;
    //Length of each example/minibatch (number of characters)
    private int exampleLength, miniBatchSize, numExamples, pointer = 0, epochSize, currEx = 0;
    private static final int THIRTY = 0, FIFTY = 1, SIXTY = 2, SEVENTY = 3, EIGHTY = 4, NINETY = 5, HOUNDRED = 6,
    HOUNDREDTWENTY = 7, BIG = 8, SMALL = 9, FOURTY = 10, MISSING = 11;
    private static final int nrSpeeds = 12;

    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
     * @throws IOException If text file cannot  be loaded
     */

    public ChSpeedIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                             char[] validCharacters, int epochSize, boolean minimized) throws IOException {
        if(!new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
        if(miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be > 0)");
        this.inputLines     = new ArrayList<>();
        this.outputLines    = new ArrayList<>();
        this.ogInput        = new ArrayList<>();
        this.ogOutput       = new ArrayList<>();
        this.inputTest      = new ArrayList<>();
        this.outputTest     = new ArrayList<>();
        this.charToIdxMap   = new HashMap<>();  //Store valid characters is a map for later use in vectorization
        this.validCharacters= validCharacters;
        this.exampleLength  = exampleLength;
        this.miniBatchSize  = miniBatchSize;
        this.epochSize      = epochSize;
        for(int i = 0; i < validCharacters.length; i++) charToIdxMap.put(validCharacters[i], i);

        //Load file and convert contents to a char[] -- ALSO filter out characters not in alphabet.
        List<String> lines = Files.readAllLines(new File(textFilePath).toPath(), textFileEncoding);
        int j = 0;
        double splitSize = lines.size() * 0.95;
        if(minimized) splitSize = 950;
        for(String s : lines){
            if(s.isEmpty()) continue;
            if(minimized && j > 1000) continue;
            j++;
            String[] inputOutput = s.split(",,,");

            if(inputOutput.length < 3) throw new IOException("Fileformat-error: can't split on ',,,' (str: " + s + ")");

            char[] inputLine = inputOutput[1].toLowerCase().toCharArray();
            int output = Integer.parseInt(inputOutput[2]);

            for(int i = 0; i < inputLine.length; i++) if(!charToIdxMap.containsKey(inputLine[i])) inputLine[i] = '!';

            if(j < splitSize){
                inputLines.add(inputLine);
                ogInput.add(inputLine);
                outputLines.add(output);
                ogOutput.add(output);
            }else{
                inputTest.add(inputLine);
                outputTest.add(output);
            }
        }
        numExamples = inputLines.size();
    }

    private int intToIdx(int speed){
        switch (speed){
            case 30:
                return THIRTY;
            case 40:
                return FOURTY;
            case 50:
                return FIFTY;
            case 60:
                return SIXTY;
            case 70:
                return SEVENTY;
            case 80:
                return EIGHTY;
            case 90:
                return NINETY;
            case 100:
                return HOUNDRED;
            case 120:
                return HOUNDREDTWENTY;
        }
        if(speed == -1) return MISSING;
        if(speed > 120) return BIG;
        return SMALL;
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
        return nrSpeeds;
    }

    public boolean hasNext() {
        return currEx < inputLines.size() && pointer < epochSize;
    }

    public boolean hasNextTest() {
        return currEx < inputTest.size();
    }

    public DataSet next() {
        return createDataSet(miniBatchSize, inputLines, outputLines);
    }

    public DataSet next(int num) {
        return createDataSet(num, inputLines, outputLines);
    }

    public DataSet createDataSet(int num, ArrayList<char[]> in, ArrayList<Integer> out){
        if(in.isEmpty()) throw new NoSuchElementException();
        int currMinibatchSize = Math.min(num, in.size() - currEx);
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        // 'f' (fortran) ordering = must for optimized custom iterator.
        INDArray input = Nd4j.create(new int[]{currMinibatchSize, validCharacters.length, exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize, nrSpeeds, exampleLength}, 'f');
//        INDArray labels = Nd4j.create(new int[]{currMinibatchSize, validCharacters.length, exampleLength*2}, 'f');

        INDArray inputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
        INDArray outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');

        for(int i=0; i < currMinibatchSize; i++) {  // Iterating each line
            char[] inputChars = in.get(currEx);
            int output = out.get(currEx);
            currEx++;
            if(inputChars == null) continue;
            outputMask.putScalar(new int[]{i, exampleLength-1}, 1f);
            // 1 = exist, 0 = should be masked. INDArray should init with zeros?
            for(int j = 0; j < inputChars.length + 1; j++)
                inputMask.putScalar(new int[]{i,j}, 1f);
//            for(int j = inputChars.length; j < (inputChars.length+outputChars.length + 1); j++)
//                outputMask.putScalar(new int[]{i,j}, 1f);
            labels.putScalar(new int[]{i, intToIdx(output), exampleLength-1}, 1f);

            for(int j = 0; j < exampleLength; j++){
                int currCharIdx = charToIdxMap.get('\n'), corrCharIdx = charToIdxMap.get('\n');
                if(inputChars.length > j) currCharIdx = charToIdxMap.get(inputChars[j]);
                input.putScalar(new int[]{i,currCharIdx,j}, 1f);
                //labels.putScalar(new int[]{i,corrCharIdx,(j+inputChars.length)}, 1.0);
            }
        }

        return new DataSet(input,labels, inputMask, outputMask);
    }

    public DataSet nextTest() {
        return createDataSet(miniBatchSize, inputTest, outputTest);
    }

    public int totalExamples() {
        return numExamples;
    }

    public int inputColumns() {
        return validCharacters.length;
    }

    public int totalOutcomes() {
        return nrSpeeds;
    }

    public void reset() {
        if(currEx < inputLines.size()){
            pointer = 0;
            return;
        }
        currEx = 0;
//        inputLines = new ArrayList<>(ogInput);
//        outputLines = new ArrayList<>(ogOutput);
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
        return totalExamples() - inputLines.size();
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