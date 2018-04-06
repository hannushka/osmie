package neural_nets.spellchecker;

import neural_nets.CharacterIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import util.ArrayUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.*;

import static util.StringUtils.getDanishCharacterSet;

/*
 * @author Alex Black
 * @author Hampus Londögård
 * @author Hannah Lindblad
 */
public class SpellCheckIterator extends CharacterIterator {
    char[] validCharacters;
    Charset textFileEncoding; //Maps each character to an index in the input/output
    Map<Character,Integer> charToIdxMap; //All characters of the input file (after filtering to only those that are valid)
    LinkedList<char[]> inputLines, outputLines;
    protected LinkedList<char[]> ogInput, ogOutput;
    protected LinkedList<char[]> inputTest, outputTest; //Length of each example/minibatch (number of characters)
    protected int exampleLength, miniBatchSize, numExamples, pointer = 0, epochSize;

    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @throws IOException If text file cannot  be loaded
     */

    public SpellCheckIterator(String textFilePath, String testFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                              int epochSize, boolean minimized, boolean useCorpus) throws IOException {
        if(!new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
        if(miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be > 0)");
        this.inputLines     = new LinkedList<>();
        this.outputLines    = new LinkedList<>();
        this.ogInput        = new LinkedList<>();
        this.ogOutput       = new LinkedList<>();
        this.inputTest      = new LinkedList<>();
        this.outputTest     = new LinkedList<>();
        this.charToIdxMap   = new HashMap<>();
        this.validCharacters= getDanishCharacterSet();
        this.exampleLength  = exampleLength;
        this.miniBatchSize  = miniBatchSize;
        this.epochSize      = epochSize;
        this.textFileEncoding = textFileEncoding;
        for(int i = 0; i < validCharacters.length; i++) charToIdxMap.put(validCharacters[i], i);
        String before = "\t", after = "\n";

        // Train-Data (streets)
        int limit = Integer.MAX_VALUE;
        if(minimized) limit = 1000;
        generateDataFromFile(textFilePath, inputLines, outputLines, limit, before, after);

        // Test-Data
        limit = Integer.MAX_VALUE;
        generateDataFromFile(testFilePath, inputTest, outputTest, limit, before, after);

        // Train-Data (corpus)
        if(useCorpus){
            LinkedList<char []> inputToMerge = new LinkedList<>(), outputToMerge = new LinkedList<>();
            generateDataFromFile(testFilePath, inputToMerge, outputToMerge, limit, "", "");
            addCorpusToData(inputToMerge, outputToMerge, before, after);
        }

        ogInput = new LinkedList<>(inputLines);
        ogOutput = new LinkedList<>(outputLines);

        numExamples = inputLines.size();
    }

    public int getNbrClasses(){
        return validCharacters.length;
    }

    public char convertIndexToCharacter( int idx ){
        return validCharacters[idx];
    }

    public int convertCharacterToIndex( char c ){
        return charToIdxMap.getOrDefault(c, 0);
    }

    public boolean hasNext() {
        return !inputLines.isEmpty() && !outputTest.isEmpty() && pointer < epochSize;
    }

    public boolean hasNextTest() {
        return !inputTest.isEmpty() && !outputTest.isEmpty();
    }

    public DataSet next() {
        return createDataSet(miniBatchSize, inputLines, outputLines);
    }

    public DataSet next(int num) {
        return createDataSet(num, inputLines, outputLines);
    }

    protected DataSet createDataSet(int num, LinkedList<char[]> in, LinkedList<char[]> out){
        if(in.isEmpty()) throw new NoSuchElementException();
        int currMinibatchSize = Math.min(num, in.size());
        currMinibatchSize = Math.min(currMinibatchSize, out.size());
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        // 'f' (fortran) ordering = must for optimized custom iterator.
        INDArray input = Nd4j.create(new int[]{currMinibatchSize, validCharacters.length, exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize, validCharacters.length, exampleLength}, 'f');
//        INDArray labels = Nd4j.create(new int[]{currMinibatchSize, validCharacters.length, exampleLength*2}, 'f');

        INDArray inputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
        INDArray outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
//        INDArray outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength*2}, 'f');

        for(int i=0; i < currMinibatchSize; i++) {  // Iterating each line
            char[] inputChars = in.removeFirst();
            char[] outputChars = out.removeFirst();
            if(inputChars == null) continue;
            pointer++;

            // 1 = exist, 0 = should be masked. INDArray should init with zeros?
            for(int j = 0; j < inputChars.length + 1; j++)
                inputMask.putScalar(new int[]{i,j}, 1f);
            for(int j = 0; j < outputChars.length + 1; j++)
                outputMask.putScalar(new int[]{i,j}, 1f);
//            for(int j = inputChars.length; j < (inputChars.length+outputChars.length + 1); j++)
//                outputMask.putScalar(new int[]{i,j}, 1f);


            for(int j = 0; j < exampleLength; j++){
                int currCharIdx = charToIdxMap.get('\n'), corrCharIdx = charToIdxMap.get('\n');
                if(inputChars.length > j) currCharIdx = charToIdxMap.get(inputChars[j]);
                if(outputChars.length > j) corrCharIdx = charToIdxMap.get(outputChars[j]);
                input.putScalar(new int[]{i,currCharIdx,j}, 1.0);
                labels.putScalar(new int[]{i,corrCharIdx,j}, 1.0);
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

    public void reset() {
        if(!inputLines.isEmpty() && !outputLines.isEmpty()){
            pointer = 0;
            return;
        }
        inputLines = new LinkedList<>(ogInput);
        outputLines = new LinkedList<>(ogOutput);
        pointer = 0;
    }

    public boolean resetSupported() {
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

    private void addCorpusToData(LinkedList<char[]> inputToMerge, LinkedList<char[]> outputToMerge,
                                 String before, String after) {
        char[] in, out;
        boolean added;
        while (!inputToMerge.isEmpty()) {
            in = inputToMerge.remove(0);
            out = outputToMerge.remove(0);
            added = false;

            for (int i = 0; i < inputToMerge.size(); i++) {
                if (in.length + 5 + inputToMerge.get(i).length < exampleLength) {
                    inputLines.add(ArrayUtils.mergeArrays(before, after, in, inputToMerge.remove(i)));
                    outputLines.add(ArrayUtils.mergeArrays(before, after, out, outputToMerge.remove(i)));
                    added = true;
                    break;
                }
            }
            if (!added) {
                inputLines.add(ArrayUtils.mergeArrays(before, after, in));
                outputLines.add(ArrayUtils.mergeArrays(before, after, out));
            }
        }
    }

    private void generateDataFromFile(String textFilePath, LinkedList<char[]> in, LinkedList<char[]> out, int limit,
                                      String before, String after) throws IOException {

        List<String> lines = Files.readAllLines(new File(textFilePath).toPath(), textFileEncoding);
        int j = 0;
        for(String s : lines){
            if(s.isEmpty() || (j > limit)) continue;
            j++;
            String[] inputOutput = s.split(",,,");

            if(inputOutput.length < 2) throw new IOException("Fileformat-error: can't split on ',,,' (str: " + s + ")");

            char[] inputLine = ArrayUtils.mergeArrays(before, after, inputOutput[0].toLowerCase().toCharArray());
            char[] outputLine = ArrayUtils.mergeArrays(before, after, inputOutput[1].toLowerCase().toCharArray());

            for(int i = 0; i < inputLine.length; i++) if(!charToIdxMap.containsKey(inputLine[i])) inputLine[i] = '!';
            for(int i = 0; i < outputLine.length; i++) if(!charToIdxMap.containsKey(outputLine[i])) outputLine[i] = '!';

            in.add(inputLine);
            out.add(outputLine);
        }
    }

}