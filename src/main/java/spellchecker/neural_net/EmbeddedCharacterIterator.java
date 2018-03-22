package spellchecker.neural_net;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.LinkedList;
import java.util.NoSuchElementException;

public class EmbeddedCharacterIterator extends CharacterIterator{
    /**
     * @param textFilePath     Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize    Number of examples per mini-batch
     * @param exampleLength    Number of characters in each input/output vector
     * @param validCharacters  Character array of valid characters. Characters not present in this array will be removed
     * @param epochSize
     * @param minimized
     * @throws IOException If text file cannot  be loaded
     */
    public EmbeddedCharacterIterator(String textFilePath, String testFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength, char[] validCharacters, int epochSize, boolean minimized) throws IOException {
        super(textFilePath, testFilePath, textFileEncoding, miniBatchSize, exampleLength, validCharacters, epochSize, minimized);
    }

    @Override
    protected DataSet createDataSet(int num, LinkedList<char[]> in, LinkedList<char[]> out){
        if(in.isEmpty()) throw new NoSuchElementException();
        int currMinibatchSize = Math.min(num, in.size());
        currMinibatchSize = Math.min(currMinibatchSize, out.size());
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        // 'f' (fortran) ordering = must for optimized custom iterator.
        INDArray input = Nd4j.create(new int[]{currMinibatchSize, 1, exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize, validCharacters.length, exampleLength}, 'f');
//        INDArray labels = Nd4j.create(new int[]{currMinibatchSize, validCharacters.length, exampleLength*2}, 'f');

        INDArray inputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
        INDArray outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
//        INDArray outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength*2}, 'f');

        for(int i=0; i < currMinibatchSize; i++) {  // Iterating each line
            char[] inputChars = in.removeFirst();
            char[] outputChars = out.removeFirst();
            if(inputChars == null) continue;

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
                input.putScalar(new int[]{i, 1, j}, currCharIdx);
                labels.putScalar(new int[]{i,corrCharIdx,j}, 1.0);
                //labels.putScalar(new int[]{i,corrCharIdx,(j+inputChars.length)}, 1.0);
            }
        }

        return new DataSet(input,labels, inputMask, outputMask);
    }
}
