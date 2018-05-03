package neural_nets.spellchecker;

import neural_nets.CharacterIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import util.ArrayUtils;
import util.DeepSpellObject;
import neural_nets.anomalies.EncoderHelper;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.*;

/*
 * @author Alex Black
 * @author Hampus Londögård
 * @author Hannah Lindblad
 */
public class SpellCheckIterator extends CharacterIterator {
    protected Charset textFileEncoding; //Maps each character to an index in the input/output
    protected Map<Character,Integer> charToIdxMap; //All characters of the input file (after filtering to only those that are valid)
    protected LinkedList<char[]> inputLines, outputLines;
    protected LinkedList<char[]> ogInput, ogOutput;
    protected int exampleLength, miniBatchSize, numExamples, pointer = 0, epochSize;
    protected boolean offset;
    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @throws IOException If text file cannot  be loaded
     */

    public SpellCheckIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                              int epochSize, boolean merge, boolean offset) throws IOException {
        if(!new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
        if(miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be > 0)");
        this.inputLines     = new LinkedList<>();
        this.outputLines    = new LinkedList<>();
        this.ogInput        = new LinkedList<>();
        this.ogOutput       = new LinkedList<>();
        this.charToIdxMap   = new HashMap<>();
        this.exampleLength  = exampleLength;
        this.miniBatchSize  = miniBatchSize;
        this.epochSize      = epochSize;
        this.textFileEncoding = textFileEncoding;
        this.offset = offset;
        charToIdxMap = EncoderHelper.getDanishCharacterSet();
        String before = "\t", after = "\n";
        generateDataFromFile(textFilePath, before, after, merge);
    }

    private void generateDataFromFile(String textFilePath, String before, String after, boolean merge) throws IOException {
        List<String> lines = Files.readAllLines(new File(textFilePath).toPath(), textFileEncoding);
        String savedInp = "", savedOut = "";
        for (String s : lines){
            if (s.isEmpty()) continue;
            s = s.toLowerCase().trim();
            if(s.length() > 96) continue;
            String[] inputOutput = s.split(",,,");
            if (inputOutput.length < 2) throw new IOException("Fileformat-error: can't split on ',,,' (str: " + s + ")");
            char[] inputLine = ArrayUtils.mergeArrays(before, after, inputOutput[0].toCharArray());
            char[] outputLine = ArrayUtils.mergeArrays(before, after, inputOutput[1].toCharArray());
            if(merge){
                if(!savedInp.isEmpty() && inputOutput[0].length() < 24){
                    inputLine = ArrayUtils.mergeArrays(before, after, savedInp.toCharArray(),
                            inputOutput[0].toCharArray());
                    outputLine = ArrayUtils.mergeArrays(before, after, savedOut.toCharArray(),
                            inputOutput[1].toCharArray());
                    savedInp = "";
                    savedOut = "";
                }
                if(savedInp.isEmpty() && inputOutput[0].length() < 24){
                    savedInp = inputOutput[0];
                    savedOut = inputOutput[1];
                    continue;
                }
            }
            for(int i = 0; i < inputLine.length; i++)  if(!charToIdxMap.containsKey(inputLine[i]))   inputLine[i] = '!';
            for(int i = 0; i < outputLine.length; i++) if(!charToIdxMap.containsKey(outputLine[i])) outputLine[i] = '!';
            inputLines.add(inputLine);
            outputLines.add(outputLine);

        }
        ogInput = new LinkedList<>(inputLines);
        ogOutput = new LinkedList<>(outputLines);
        numExamples = inputLines.size();
    }



    // dimension 0 = number of examples in minibatch
    // dimension 1 = size of each vector (i.e., number of characters)
    // dimension 2 = length of each time series/example
    // 'f' (fortran) ordering = must for optimized custom iterator.
    protected DataSet createDataSet(int num) {
        int currMinibatchSize;
        if (inputLines.isEmpty()) throw new NoSuchElementException();
        currMinibatchSize = Math.min(num, inputLines.size());
        INDArray input, labels, inputMask, outputMask;

        if(!offset){
            input = Nd4j.create(new int[]{currMinibatchSize, charToIdxMap.size(), exampleLength}, 'f');
            labels = Nd4j.create(new int[]{currMinibatchSize, charToIdxMap.size(), exampleLength}, 'f');

            // 1 = exist, 0 = should be masked
            inputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
            outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
        }else{
            input = Nd4j.create(new int[]{currMinibatchSize, charToIdxMap.size(), exampleLength*2}, 'f');
            labels = Nd4j.create(new int[]{currMinibatchSize, charToIdxMap.size(), exampleLength*2}, 'f');
            inputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength*2}, 'f');
            outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength*2}, 'f');
        }

        for (int i = 0; i < currMinibatchSize; i++) {
           char[] inputChars = inputLines.removeFirst();
           char[] outputChars = outputLines.removeFirst();

            if(inputChars == null) continue;
            pointer++;

            for(int j = 0; j < inputChars.length + 1; j++)
                inputMask.putScalar(new int[]{i,j}, 1f);

            if(!offset){
                for(int j = 0; j < (outputChars.length + 1); j++)
                    outputMask.putScalar(new int[]{i,j}, 1f);
            }else{
                for(int j = inputChars.length; j < (inputChars.length + outputChars.length + 1); j++)
                    outputMask.putScalar(new int[]{i,j}, 1f);
            }

            for(int j = 0; j < exampleLength; j++){
                int currCharIdx = charToIdxMap.get('\n'), corrCharIdx = charToIdxMap.get('\n');
                if(inputChars.length > j) currCharIdx = charToIdxMap.get(inputChars[j]);
                if(outputChars.length > j) corrCharIdx = charToIdxMap.get(outputChars[j]);
                input.putScalar(new int[]{i,currCharIdx,j}, 1.0);
                if(!offset) labels.putScalar(new int[]{i,corrCharIdx,j}, 1.0);
                else labels.putScalar(new int[]{i,corrCharIdx,j + inputChars.length}, 1.0);
            }
        }
        return new DataSet(input, labels, inputMask, outputMask);
    }

    public boolean hasNext() {
        return !inputLines.isEmpty() && !outputLines.isEmpty() && pointer < epochSize;
    }

    public DataSet next() {
        return createDataSet(miniBatchSize);
    }

    public DataSet next(int num) {
        return createDataSet(num);
    }

    public int totalExamples() {
        return numExamples;
    }

    public int inputColumns() {
        return charToIdxMap.size();
    }

    public int totalOutcomes() {
        return inputColumns();
    }

    public void reset() {
        if (!inputLines.isEmpty() && !outputLines.isEmpty()) {
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

    public char convertIndexToCharacter(int idx) {
        for (Character c : charToIdxMap.keySet()) {
            if (charToIdxMap.get(c) == idx)
                return c;
        }
        return '!';
    }

    public DataSet createDataSetFromDSO(DeepSpellObject obj){
        String labelStr = obj.correctName;
        char[] label = ArrayUtils.mergeArrays("\t", "\n", labelStr.toCharArray());
        char[] inp;
        for(int i = 0; i < label.length; i++) if(!charToIdxMap.containsKey(label[i])) label[i] = '!';
        for(String suggestion : obj.corrections){
            inp = ArrayUtils.mergeArrays("\t", "\n", suggestion.toCharArray());
            for(int i = 0; i < inp.length; i++) if(!charToIdxMap.containsKey(inp[i])) inp[i] = '!';
            inputLines.add(inp);
            outputLines.add(label);
        }
        return createDataSet(Integer.MAX_VALUE);
    }

    public DataSet createDataSetForString(String name){
        char[] inp = ArrayUtils.mergeArrays("\t", "\n", name.toCharArray());
        for (int i = 0; i < inp.length; i++) if(!charToIdxMap.containsKey(inp[i])) inp[i] = '!';
        INDArray input = Nd4j.create(new int[]{1, charToIdxMap.size(), exampleLength}, 'f');
        INDArray inputMask = Nd4j.zeros(new int[]{1, exampleLength}, 'f');
        INDArray outputMask = Nd4j.ones(new int[]{1, exampleLength});
        char[] inputChars = inp;
        for (int j = 0 ; j < inputChars.length + 1 ; j++)
            inputMask.putScalar(new int[]{0,j}, 1f);
        for (int j = 0 ; j < exampleLength ; j++){
            int currCharIdx = charToIdxMap.get('\n');
            if (inputChars.length > j) currCharIdx = charToIdxMap.get(inputChars[j]);
            input.putScalar(new int[]{0,currCharIdx,j}, 1.0);
        }
        return new DataSet(input, null, inputMask, outputMask);
    }
}