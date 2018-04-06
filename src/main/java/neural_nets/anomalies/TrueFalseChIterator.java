package neural_nets.anomalies;

import neural_nets.CharacterIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import neural_nets.spellchecker.SpellCheckIterator;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.*;

import static util.StringUtils.getDanishCharacterSet;

public class TrueFalseChIterator extends CharacterIterator {
    char[] validCharacters;
    Charset textFileEncoding; //Maps each character to an index in the input/output
    Map<Character,Integer> charToIdxMap; //All characters of the input file (after filtering to only those that are valid)
    LinkedList<char[]> inputLines, outputLines;
    protected LinkedList<char[]> ogInput, ogOutput;
    protected LinkedList<char[]> inputTest, outputTest; //Length of each example/minibatch (number of characters)
    protected int exampleLength, miniBatchSize, numExamples, pointer = 0, epochSize;
    List<Boolean> outputTF, testOutputTF;
    List<List<String>> inputTF, testTF;

    public TrueFalseChIterator(String file, Charset encoding, int miniBatchSize, int sequenceLength, int epochSize,
                               boolean minimized) throws IOException {
            if(!new File(file).exists()) throw new IOException("Could not access file (does not exist): " + file);
            if(miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be > 0)");
            this.outputTF = new ArrayList<>();
            this.testOutputTF = new ArrayList<>();
            this.inputTF = new ArrayList<>();
            this.testTF = new ArrayList<>();

            this.charToIdxMap   = new HashMap<>();
            this.validCharacters= getDanishCharacterSet();
            this.exampleLength  = sequenceLength;
            this.miniBatchSize  = miniBatchSize;
            this.epochSize      = epochSize;
            this.textFileEncoding = encoding;
            for(int i = 0; i < validCharacters.length; i++) charToIdxMap.put(validCharacters[i], i);
            String before = "\t", after = "\n";

            // Train-Data (streets)
            generateDataFromFile(file, before, after);

            ogInput = new LinkedList<>(inputLines);
            ogOutput = new LinkedList<>(outputLines);
            numExamples = inputLines.size();
        }

        private void generateDataFromFile(String textFilePath, String before, String after) throws IOException {
            List<String> lines = Files.readAllLines(new File(textFilePath).toPath(), textFileEncoding);
            int j = 0, limit = (int) (lines.size() * 0.95f);

            for(String s : lines){
                if(s.isEmpty() || (j > limit)) continue;
                j++;
                String[] inputOutput = s.split(",,,");

                if(inputOutput.length < 7) throw new IOException("Fileformat-error: can't split on ',,,' (str: " + s + ")");

                List<String> values = new ArrayList<>();
                for(String val : inputOutput) values.add(val.toLowerCase().trim());
                values.remove(0);
                // values = [val,val,val,val,name] ish.
//                char[] outputLine = new char[1];


                //for(int i = 0; i < inputLine.length; i++) if(!charToIdxMap.containsKey(inputLine[i])) inputLine[i] = '!';
                //int val = inputOutput[0].toLowerCase().trim().equals(inputOutput[1].toLowerCase().trim()) ? 1 : 0;

                if(j < limit){
                }else{
                }
            }
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
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize, 1, exampleLength}, 'f');
        INDArray inputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
        INDArray outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');

        for(int i=0; i < currMinibatchSize; i++) {  // Iterating each line
            char[] inputChars = in.removeFirst();
            char[] outputChars = out.removeFirst();
            if(inputChars == null) continue;
            pointer++;

            // 1 = exist, 0 = should be masked. INDArray should init with zeros?
            for(int j = 0; j < inputChars.length + 1; j++)
                inputMask.putScalar(new int[]{i,j}, 1f);

            outputMask.putScalar(new int[]{i, exampleLength-1}, 1f);
            labels.putScalar(new int[]{i, 0, exampleLength-1}, (float) outputChars[0]);

            for(int j = 0; j < exampleLength; j++){
                int currCharIdx = charToIdxMap.get('\n');
                if(inputChars.length > j) currCharIdx = charToIdxMap.get(inputChars[j]);
                input.putScalar(new int[]{i,currCharIdx,j}, 1f);
            }
        }

        return new DataSet(input,labels, inputMask, outputMask);
    }

    @Override
    public DataSet next(int num) {
        return null;
    }

    @Override
    public int totalExamples() {
        return 0;
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 1;
    }

    @Override
    public char convertIndexToCharacter(int idx) {
        return 0;
    }

    @Override
    public int getNbrClasses() {
        return 0;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public void reset() {

    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public int cursor() {
        return 0;
    }

    @Override
    public int numExamples() {
        return 0;
    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public DataSet next() {
        return null;
    }
}
