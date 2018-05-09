package neural_nets.anomalies;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import util.ArrayUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;

public class AnomaliesPredictIterator extends AnomaliesIterator {
    public AnomaliesPredictIterator(String file, Charset encoding, int miniBatchSize, int sequenceLength, int epochSize) throws IOException {
        super(file, encoding, miniBatchSize, sequenceLength, epochSize);
    }

    @Override
    protected void generateDataFromFile(String textFilePath, String before, String after) throws IOException {
        List<String> lines = Files.readAllLines(new File(textFilePath).toPath(), textFileEncoding);
        for (String s : lines){
            if (s.isEmpty()) continue;
            String[] values = s.split(",,,", -1);
            if (values.length < 5) throw new IOException("Fileformat-error: can't split on ',,,' (str: " + s + ")");
            //name,,,maxspeed,,,surface,,highway,,,0/1

            if (values[1].isEmpty()) continue;

            long id = Long.parseLong(values[0]);
            char[] name = ArrayUtils.mergeArrays(before, after, values[1].toLowerCase().toCharArray());
            int maxSpeedClass = speedToIdxMap.get(EncoderHelper.getSpeedClass(values[2]));
            int surfaceClass =surfaceToIdxMap.get(EncoderHelper.getSurfaceClass(values[3]));
            int highwayClass = highwayToIdxMap.get(EncoderHelper.getHighwayClass(values[4]));

            for (int i = 0; i < name.length; i++) if(!charToIdxMap.containsKey(name[i])) name[i] = '!';
            inputLines.add(new DataContainer(id, name, maxSpeedClass, highwayClass, surfaceClass));
        }
        ogInput = new LinkedList<>(inputLines);
        numExamples = inputLines.size();
    }

    @Override
    protected DataSet createDataSet(int num) {
        if(inputLines.isEmpty()) throw new NoSuchElementException();
        int currMinibatchSize = Math.min(num, inputLines.size());
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        // 'f' (fortran) ordering = must for optimized custom iterator.
        INDArray input = Nd4j.create(new int[]{currMinibatchSize, 1, exampleLength}, 'f');
        INDArray inputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');
        INDArray outputMask = Nd4j.zeros(new int[]{currMinibatchSize, exampleLength}, 'f');

        for(int i=0; i < currMinibatchSize; i++) {  // Iterating each line
            DataContainer cont = inputLines.removeFirst();
            lastBatchContainers.add(cont);
            char[] inputChars =  cont.inputLine;
            if(inputChars == null) continue;

            // 1 = exist, 0 = should be masked. INDArray should init with zeros?
            for(int j = 0; j < Math.min(exampleLength, inputChars.length + 1); j++)
                inputMask.putScalar(new int[]{i,j}, 1f);

            outputMask.putScalar(new int[]{i, exampleLength-1}, 1f);

            //Add name
            for(int j = 1; j < Math.min(inputChars.length, exampleLength - AnomaliesIterator.nbrOfTags); j++){
                int currCharIdx = charToIdxMap.get('\n');
                if(inputChars.length > j) currCharIdx = charToIdxMap.get(inputChars[inputChars.length-j]);
                input.putScalar(new int[]{i,0,j}, currCharIdx);
            }

            //Add tags
            input.putScalar(new int[]{i, 0, exampleLength-3}, cont.maxSpeedClass);
            input.putScalar(new int[]{i, 0, exampleLength-2}, cont.highwayClass);
            input.putScalar(new int[]{i, 0, exampleLength-1}, cont.surfaceClass);
        }
        return new DataSet(input, null, inputMask, outputMask);
    }

    public boolean hasNext() {
        return !inputLines.isEmpty();
    }
}
