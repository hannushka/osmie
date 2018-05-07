package neural_nets.spellchecker;

import neural_nets.CharacterIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import neural_nets.anomalies.EncoderHelper;

import java.io.*;
import java.nio.charset.Charset;
import java.util.*;

/*
 * @author Hampus Londögård
 */
public class ShakeSpeareIterator extends CharacterIterator {
    protected Charset textFileEncoding; //Maps each character to an index in the input/output
    protected Map<Character,Integer> charToIdxMap; //All characters of the input file (after filtering to only those that are valid)
    protected int exampleLength, miniBatchSize, numExamples, pointer = 0, epochSize;
    protected String currentLine;
    protected BufferedReader bufferedReader;

    public ShakeSpeareIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                              int epochSize) throws IOException {
        if(!new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
        if(miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be > 0)");
        this.charToIdxMap   = new HashMap<>();
        this.exampleLength  = exampleLength;
        this.miniBatchSize  = miniBatchSize;
        this.epochSize      = epochSize;
        this.textFileEncoding = textFileEncoding;
        charToIdxMap = EncoderHelper.getDanishCharacterSet();
        bufferedReader = new BufferedReader(new InputStreamReader(
                new FileInputStream(textFilePath), "UTF-8"));
    }

    protected DataSet createDataSet(int num) throws IOException {
        INDArray input, labels;
        input = Nd4j.create(new int[]{num, charToIdxMap.size(), exampleLength}, 'f');
        labels = Nd4j.create(new int[]{num, charToIdxMap.size(), exampleLength}, 'f');
        int size;
        StringBuilder tempInp;

        for (int i = 0; i < num; i++) {
            size = 0;
            tempInp = new StringBuilder();
            currentLine = bufferedReader.readLine();
//            System.out.println(currentLine);

            while(currentLine != null && (size + currentLine.length()) < exampleLength){
                size += currentLine.length();
                tempInp.append(currentLine);
                currentLine = bufferedReader.readLine();
            }

            size = exampleLength - size;
            if(currentLine != null) tempInp.append(currentLine, 0, size);
            String[] split = tempInp.toString().split("\\.");
            tempInp = new StringBuilder();
            for(int j = 0; j < split.length - 1; j++) tempInp.append(split[j]);
            char[] inputChars = tempInp.toString().toCharArray();
            pointer++;

            for(int j = 1; j < exampleLength; j++){
                int currCharIdx = charToIdxMap.get('\n'), corrCharIdx = charToIdxMap.get('\n');
                if(inputChars.length > j){
                    currCharIdx = charToIdxMap.getOrDefault(inputChars[j-1], charToIdxMap.get('!'));
                    corrCharIdx = charToIdxMap.getOrDefault(inputChars[j], charToIdxMap.get('!'));
                }
                input.putScalar(new int[]{i,currCharIdx,j-1}, 1.0);
                labels.putScalar(new int[]{i,corrCharIdx,j-1}, 1.0);
            }
            labels.putScalar(new int[]{i,charToIdxMap.get('\n'),exampleLength-1}, 1.0);
        }

        return new DataSet(input, labels);
    }

    public boolean hasNext() {
        return pointer < epochSize;
    }

    public DataSet next() {
        try {
            return createDataSet(miniBatchSize);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public DataSet next(int num) {
        try {
            return createDataSet(num);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
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
        pointer = 0;
    }

    public boolean resetSupported() {
        return true;
    }

    public int batch() {
        return miniBatchSize;
    }

    public int cursor() {
        return totalExamples() - pointer;
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
}