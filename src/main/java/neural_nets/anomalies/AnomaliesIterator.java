package neural_nets.anomalies;

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

public class AnomaliesIterator extends CharacterIterator {
    char[] validCharacters;
    Charset textFileEncoding; //Maps each character to an index in the input/output
    Map<Character,Integer> charToIdxMap; //All characters of the input file (after filtering to only those that are valid)
    LinkedList<char[]> inputLines;
    LinkedList<Integer> outputLines;
    protected LinkedList<char[]> ogInput;
    LinkedList<Integer> ogOutput;
    protected int exampleLength, miniBatchSize, numExamples, pointer = 0, epochSize;

    public AnomaliesIterator(String file, Charset encoding, int miniBatchSize, int sequenceLength, int epochSize) throws IOException {
            if(!new File(file).exists()) throw new IOException("Could not access file (does not exist): " + file);
            if(miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be > 0)");
            this.inputLines     = new LinkedList<>();
            this.outputLines    = new LinkedList<>();
            this.charToIdxMap   = new HashMap<>();
            this.validCharacters= getDanishCharacterSet();
            this.exampleLength  = sequenceLength;
            this.miniBatchSize  = miniBatchSize;
            this.epochSize      = epochSize;
            this.textFileEncoding = encoding;
            for(int i = 0; i < validCharacters.length; i++) charToIdxMap.put(validCharacters[i], i);
            String before = "\t", after = "\n";
            generateDataFromFile(file, before, after);
        }

        private void generateDataFromFile(String textFilePath, String before, String after) throws IOException {
            List<String> lines = Files.readAllLines(new File(textFilePath).toPath(), textFileEncoding);
            for (String s : lines){
                if (s.isEmpty()) continue;
                String[] values = s.split(",,,");

                if (values.length < 7) throw new IOException("Fileformat-error: can't split on ',,,' (str: " + s + ")");

                char[] inputLine = ArrayUtils.mergeArrays(before, after, values[0].toLowerCase().toCharArray());
                char[] tmpOutput = values[values.length-1].toLowerCase().toCharArray();
                int outputLine = tmpOutput.equals(inputLine) ? 0 : 1;

                for (int i = 0; i < inputLine.length; i++) if(!charToIdxMap.containsKey(inputLine[i])) inputLine[i] = '!';
                inputLines.add(inputLine);
                outputLines.add(outputLine);
            }
            ogInput = new LinkedList<>(inputLines);
            ogOutput = new LinkedList<>(outputLines);
            numExamples = inputLines.size();
        }

    protected DataSet createDataSet(int num) {
        if(inputLines.isEmpty()) throw new NoSuchElementException();
        int currMinibatchSize = Math.min(num, inputLines.size());
        currMinibatchSize = Math.min(currMinibatchSize, outputLines.size());
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        // 'f' (fortran) ordering = must for optimized custom iterator.
        INDArray input = Nd4j.create(new int[]{currMinibatchSize, validCharacters.length, exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize, 2, exampleLength}, 'f');
        INDArray inputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
        INDArray outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');

        for(int i=0; i < currMinibatchSize; i++) {  // Iterating each line
            char[] inputChars = inputLines.removeFirst();
            int outputClass = outputLines.removeFirst();
            if(inputChars == null) continue;
            pointer++;

            // 1 = exist, 0 = should be masked. INDArray should init with zeros?
            for(int j = 0; j < inputChars.length + 1; j++)
                inputMask.putScalar(new int[]{i,j}, 1f);

            outputMask.putScalar(new int[]{i, exampleLength-1}, 1f);
            labels.putScalar(new int[]{i, outputClass, exampleLength-1}, 1f);

            for(int j = 0; j < exampleLength; j++){
                int currCharIdx = charToIdxMap.get('\n');
                if(inputChars.length > j) currCharIdx = charToIdxMap.get(inputChars[j]);
                input.putScalar(new int[]{i,currCharIdx,j}, 1f);
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

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

}
