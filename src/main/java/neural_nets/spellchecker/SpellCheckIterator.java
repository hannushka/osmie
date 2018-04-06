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
    protected int exampleLength, miniBatchSize, numExamples, pointer = 0, epochSize;

    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @throws IOException If text file cannot  be loaded
     */

    public SpellCheckIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                              int epochSize) throws IOException {
        if(!new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
        if(miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be > 0)");
        this.inputLines     = new LinkedList<>();
        this.outputLines    = new LinkedList<>();
        this.ogInput        = new LinkedList<>();
        this.ogOutput       = new LinkedList<>();
        this.charToIdxMap   = new HashMap<>();
        this.validCharacters= getDanishCharacterSet();
        this.exampleLength  = exampleLength;
        this.miniBatchSize  = miniBatchSize;
        this.epochSize      = epochSize;
        this.textFileEncoding = textFileEncoding;
        for(int i = 0; i < validCharacters.length; i++) charToIdxMap.put(validCharacters[i], i);
        String before = "\t", after = "\n";
        generateDataFromFile(textFilePath, before, after);
    }


    private void generateDataFromFile(String textFilePath, String before, String after) throws IOException {

        List<String> lines = Files.readAllLines(new File(textFilePath).toPath(), textFileEncoding);
        int j = 0;
        for (String s : lines){
            if (s.isEmpty()) continue;
            j++;
            String[] inputOutput = s.split(",,,");

            if (inputOutput.length < 2) throw new IOException("Fileformat-error: can't split on ',,,' (str: " + s + ")");

            char[] inputLine = ArrayUtils.mergeArrays(before, after, inputOutput[0].toLowerCase().toCharArray());
            char[] outputLine = ArrayUtils.mergeArrays(before, after, inputOutput[1].toLowerCase().toCharArray());

            for (int i = 0; i < inputLine.length; i++) if(!charToIdxMap.containsKey(inputLine[i])) inputLine[i] = '!';
            for (int i = 0; i < outputLine.length; i++) if(!charToIdxMap.containsKey(outputLine[i])) outputLine[i] = '!';

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
        currMinibatchSize = Math.min(currMinibatchSize, outputLines.size());


        INDArray input = Nd4j.create(new int[]{currMinibatchSize, validCharacters.length, exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize, validCharacters.length, exampleLength}, 'f');

        // 1 = exist, 0 = should be masked
        INDArray inputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
        INDArray outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');

        for (int i = 0; i < currMinibatchSize; i++) {
           char[] inputChars = inputLines.removeFirst();
           char[] outputChars = outputLines.removeFirst();

            if(inputChars == null) continue;
            pointer++;

            for(int j = 0; j < inputChars.length + 1; j++)
                inputMask.putScalar(new int[]{i,j}, 1f);
            for(int j = 0; j < outputChars.length + 1; j++)
                outputMask.putScalar(new int[]{i,j}, 1f);
            for(int j = 0; j < exampleLength; j++){
                int currCharIdx = charToIdxMap.get('\n'), corrCharIdx = charToIdxMap.get('\n');
                if(inputChars.length > j) currCharIdx = charToIdxMap.get(inputChars[j]);
                if(outputChars.length > j) corrCharIdx = charToIdxMap.get(outputChars[j]);
                input.putScalar(new int[]{i,currCharIdx,j}, 1.0);
                labels.putScalar(new int[]{i,corrCharIdx,j}, 1.0);
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
        return validCharacters.length;
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

    public int getNbrClasses(){
        return validCharacters.length;
    }

    public char convertIndexToCharacter( int idx ){
        return validCharacters[idx];
    }

    public int convertCharacterToIndex( char c ){
        return charToIdxMap.getOrDefault(c, 0);
    }
}