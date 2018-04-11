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

public class AnomaliesIterator extends CharacterIterator {
    Charset textFileEncoding; //Maps each character to an index in the input/output
    LinkedList<DataContainer> inputLines;
    LinkedList<Integer> outputLines;
        protected LinkedList<DataContainer> ogInput;
    LinkedList<Integer> ogOutput;
    protected int exampleLength, miniBatchSize, numExamples, pointer = 0, epochSize;
    protected int alphabetSize;

    Map<Character,Integer> charToIdxMap; //All characters of the input file (after filtering to only those that are valid)
    Map<String, Integer> speedToIdxMap;
    Map<String, Integer> highwayToIdxMap;
    Map<String, Integer> surfaceToIdxMap;

    private static int nbrOfTags = 3;

    public AnomaliesIterator(String file, Charset encoding, int miniBatchSize, int sequenceLength, int epochSize) throws IOException {
            if(!new File(file).exists()) throw new IOException("Could not access file (does not exist): " + file);
            if(miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be > 0)");
            this.inputLines     = new LinkedList<>();
            this.outputLines    = new LinkedList<>();
            this.charToIdxMap   = new HashMap<>();
            this.exampleLength  = sequenceLength + nbrOfTags;
            this.miniBatchSize  = miniBatchSize;
            this.epochSize      = epochSize;
            this.textFileEncoding = encoding;

            charToIdxMap = EncoderHelper.getDanishCharacterSet();
            alphabetSize = charToIdxMap.size();
            speedToIdxMap = EncoderHelper.getMaxSpeedMap(alphabetSize);
            alphabetSize += speedToIdxMap.size();
            highwayToIdxMap = EncoderHelper.getHighwayMap(alphabetSize);
            alphabetSize += highwayToIdxMap.size();
            surfaceToIdxMap = EncoderHelper.getSurfaceMap(alphabetSize);
            alphabetSize += surfaceToIdxMap.size();

            System.out.println("Alphabet size is " + alphabetSize);

            String before = "\t", after = "\n";
            generateDataFromFile(file, before, after);
        }

        private void generateDataFromFile(String textFilePath, String before, String after) throws IOException {
            List<String> lines = Files.readAllLines(new File(textFilePath).toPath(), textFileEncoding);
            for (String s : lines){
                if (s.isEmpty()) continue;
                String[] values = s.split(",,,");
                if (values.length < 7) throw new IOException("Fileformat-error: can't split on ',,,' (str: " + s + ")");
                //name,,,maxspeed,,,surface,,sidewalk,,,highway,,,oneway,,,0/1

                char[] name = ArrayUtils.mergeArrays(before, after, values[0].toLowerCase().toCharArray());
                int maxSpeedClass = speedToIdxMap.get(EncoderHelper.getSpeedClass(values[1]));
                int surfaceClass =surfaceToIdxMap.get(EncoderHelper.getSurfaceClass(values[2]));
                int highwayClass = highwayToIdxMap.get(EncoderHelper.getHighwayClass(values[4]));

                Integer result = Integer.parseInt(values[values.length-1]);

                for (int i = 0; i < name.length; i++) if(!charToIdxMap.containsKey(name[i])) name[i] = '!';
                inputLines.add(new DataContainer(name, maxSpeedClass, highwayClass, surfaceClass));
                outputLines.add(result);
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
        INDArray input = Nd4j.create(new int[]{currMinibatchSize, 1, exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize, 2, exampleLength}, 'f');
        INDArray inputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
        INDArray outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');

        for(int i=0; i < currMinibatchSize; i++) {  // Iterating each line
            DataContainer cont = inputLines.removeFirst();
            char[] inputChars =  cont.inputLine;
            int outputClass = outputLines.removeFirst();
            if(inputChars == null) continue;
            pointer++;

            // 1 = exist, 0 = should be masked. INDArray should init with zeros?
            for(int j = 0; j < Math.min(exampleLength, inputChars.length + 1); j++)
                inputMask.putScalar(new int[]{i,j}, 1f);

            outputMask.putScalar(new int[]{i, exampleLength-1}, 1f);

            labels.putScalar(new int[]{i, outputClass, exampleLength-1}, 1f);

            //Add name
            for(int j = 1; j < Math.min(inputChars.length, exampleLength - nbrOfTags); j++){
                int currCharIdx = charToIdxMap.get('\n');
                if(inputChars.length > j) currCharIdx = charToIdxMap.get(inputChars[inputChars.length-j]);
                input.putScalar(new int[]{i,0,j}, currCharIdx);
            }

            //Add tags
            input.putScalar(new int[]{i, 0, exampleLength-3}, cont.maxSpeedClass);
            input.putScalar(new int[]{i, 0, exampleLength-2}, cont.highwayClass);
            input.putScalar(new int[]{i, 0, exampleLength-1}, cont.surfaceClass);
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
        return alphabetSize;
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

    public char convertIndexToCharacter(int idx) {
        for (Character c : charToIdxMap.keySet()) {
            if (charToIdxMap.get(c) == idx)
                return c;
        }
        return '!';
    }

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    private class DataContainer {
        char[] inputLine;
        int maxSpeedClass, highwayClass, surfaceClass;

        public DataContainer(char[] inputLine, int maxSpeedClass, int highwayClass, int surfaceClass) {
            this.inputLine = inputLine;
            this.maxSpeedClass = maxSpeedClass;
            this.highwayClass = highwayClass;
            this.surfaceClass = surfaceClass;
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("name: ").append(new String(inputLine).trim()).append(", ");
            sb.append("maxspeed: ").append(maxSpeedClass).append(", ");
            sb.append("surface: ").append(surfaceClass).append(", ");
            sb.append("highway: ").append(highwayClass).append("\n");
            return sb.toString();
        }
    }

}
