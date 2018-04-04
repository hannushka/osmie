package spellchecker.neural_net;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import util.ArrayUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;

import static util.StringUtils.getDanishCharacterSet;

public class TrueFalseChIterator extends CharacterIterator {
    public TrueFalseChIterator(String file, Charset encoding, int miniBatchSize, int sequenceLength, int epochSize,
                               boolean minimized) throws IOException {
            if(!new File(file).exists()) throw new IOException("Could not access file (does not exist): " + file);
            if(miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be > 0)");
            this.inputLines     = new LinkedList<>();
            this.outputLines    = new LinkedList<>();
            this.ogInput        = new LinkedList<>();
            this.ogOutput       = new LinkedList<>();
            this.inputTest      = new LinkedList<>();
            this.outputTest     = new LinkedList<>();
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

                if(inputOutput.length < 2) throw new IOException("Fileformat-error: can't split on ',,,' (str: " + s + ")");

                char[] inputLine = ArrayUtils.mergeArrays(before, after, inputOutput[0].toLowerCase().toCharArray());
                char[] outputLine = new char[1];

                for(int i = 0; i < inputLine.length; i++) if(!charToIdxMap.containsKey(inputLine[i])) inputLine[i] = '!';
                int val = inputOutput[0].toLowerCase().trim().equals(inputOutput[1].toLowerCase().trim()) ? 1 : 0;
                outputLine[0] = (char) val;

                if(j < limit){
                    inputLines.add(inputLine);
                    outputLines.add(outputLine);
                }else{
                    inputTest.add(inputLine);
                    outputTest.add(outputLine);
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
    public int totalOutcomes() {
        return 1;
    }
}
