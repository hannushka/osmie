package spellchecker.neural_net;

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
public class CharacterIterator implements DataSetIterator {
    protected char[] validCharacters;
    //Maps each character to an index in the input/output
    protected Map<Character,Integer> charToIdxMap;
    //All characters of the input file (after filtering to only those that are valid)
    protected LinkedList<char[]> inputLines, outputLines;
    protected LinkedList<char[]> ogInput, ogOutput;
    protected LinkedList<char[]> inputTest, outputTest;
    //Length of each example/minibatch (number of characters)
    protected int exampleLength, miniBatchSize, numExamples, pointer = 0, epochSize;
    //Size of each minibatch (number of examples) <-- This is each word, so words here.

    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
     * @throws IOException If text file cannot  be loaded
     */

    public CharacterIterator(String textFilePath, String testFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                             char[] validCharacters, int epochSize, boolean minimized) throws IOException {
        if(!new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
        if(miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be > 0)");
        this.inputLines     = new LinkedList<>();
        this.outputLines    = new LinkedList<>();
        this.ogInput        = new LinkedList<>();
        this.ogOutput       = new LinkedList<>();
        this.inputTest      = new LinkedList<>();
        this.outputTest     = new LinkedList<>();
        this.charToIdxMap   = new HashMap<>();  //Store valid characters is a map for later use in vectorization
        this.validCharacters= validCharacters;
        this.exampleLength  = exampleLength;
        this.miniBatchSize  = miniBatchSize;
        this.epochSize      = epochSize;
        for(int i = 0; i < validCharacters.length; i++) charToIdxMap.put(validCharacters[i], i);

        //Load file and convert contents to a char[] -- ALSO filter out characters not in alphabet.
        List<String> lines = Files.readAllLines(new File(textFilePath).toPath(), textFileEncoding);
        int j = 0;
        for(String s : lines){
            if(s.isEmpty() || (minimized && j > 1000)) continue;
            j++;
            String[] inputOutput = s.split(",,,");

            if(inputOutput.length < 2) throw new IOException("Fileformat-error: can't split on ',,,' (str: " + s + ")");

            char[] inputLine = (inputOutput[0].toLowerCase() + "\n").toLowerCase().toCharArray();
            char[] outputLine = ("\t" + inputOutput[1].toLowerCase() + "\n").toCharArray();  // Start and end character

            for(int i = 0; i < inputLine.length; i++) if(!charToIdxMap.containsKey(inputLine[i])) inputLine[i] = '!';
            for(int i = 0; i < outputLine.length; i++) if(!charToIdxMap.containsKey(outputLine[i])) outputLine[i] = '!';

            inputLines.add(inputLine);
            outputLines.add(outputLine);
        }

        lines = Files.readAllLines(new File(testFilePath).toPath(), textFileEncoding);
        for(String s : lines){
            if(s.isEmpty()) continue;
            String[] inputOutput = s.split(",,,");

            if(inputOutput.length < 2) throw new IOException("Fileformat-error: can't split on ',,,' (str: " + s + ")");

            char[] inputLine = (inputOutput[0] + "\n").toLowerCase().toCharArray();
            char[] outputLine = ("\t" + inputOutput[1].toLowerCase() + "\n").toCharArray();  // Start and end character

            for(int i = 0; i < inputLine.length; i++) if(!charToIdxMap.containsKey(inputLine[i])) inputLine[i] = '!';
            for(int i = 0; i < outputLine.length; i++) if(!charToIdxMap.containsKey(outputLine[i])) outputLine[i] = '!';

            inputTest.add(inputLine);
            outputTest.add(outputLine);
        }

        ArrayList<char []> inputToMerge = new ArrayList<>(), outputToMerge = new ArrayList<>();
        if(!minimized){
            lines = Files.readAllLines(new File("data/korpus_freq_dict.txt.mini.noised").toPath(), textFileEncoding);
            for(String s : lines) {         // TODO check if \n is included or not!!
                if(s.isEmpty()) continue;  // TODO 10k as limit as in the KERAS example..!
                String[] inputOutput = s.split(",,,");
                if(inputOutput.length != 2){
                    throw new IOException("Fileformat error: '<old> <nbr>' is to be used compared to " + s);
                }

                char[] inputLine = inputOutput[0].toLowerCase().toCharArray();
                char[] outputLine = inputOutput[1].toLowerCase().toCharArray();  // Start and end character
                if(inputLine.length > 49 || outputLine.length > 49) continue;

                for(int i = 0; i < inputLine.length; i++) if(!charToIdxMap.containsKey(inputLine[i])) inputLine[i] = '!';
                for(int i = 0; i < outputLine.length; i++) if(!charToIdxMap.containsKey(outputLine[i])) outputLine[i] = '!';

                inputToMerge.add(inputLine);
                outputToMerge.add(outputLine);
            }
        }
        char[] in, out;
        boolean added;
        while(!inputToMerge.isEmpty()){
            in = inputToMerge.remove(0);
            out = outputToMerge.remove(0);
            added = false;

            for(int i = 0; i < inputToMerge.size(); i++){
                if(in.length + 1 + inputToMerge.get(i).length < exampleLength){
                    inputLines.add(Helper.mergeInArrays(in, inputToMerge.remove(i), ' '));
                    outputLines.add(Helper.mergeOutArrays(out, outputToMerge.remove(i), ' '));
                    added = true;
                    break;
                }
            }

            if(!added){
                inputLines.add(in);
                outputLines.add(out);
            }
        }
        ogInput = new LinkedList<>(inputLines);
        ogOutput = new LinkedList<>(outputLines);

        numExamples = inputLines.size();
    }

    protected CharacterIterator() { }

    /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
    private static char[] getMinimalCharacterSet(){
        List<Character> validChars = new LinkedList<>();
        for(char c='a'; c<='z'; c++) validChars.add(c);
        for(char c='A'; c<='Z'; c++) validChars.add(c);
        for(char c='0'; c<='9'; c++) validChars.add(c);
        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
        for(char c : temp) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i=0;
        for(Character c : validChars) out[i++] = c;
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

    public int totalOutcomes() {
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