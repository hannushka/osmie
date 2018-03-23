package spellchecker.neural_net;

import org.nd4j.linalg.api.ndarray.INDArray;
import speed.neural_net.ChSpeedIterator;

import java.nio.charset.Charset;

public class Helper {
    public static CharacterIterator getCharacterIterator(int miniBatchSize, int sequenceLength, int epochSize,
                                                         boolean minimized) throws Exception {
        String fileLocation = "data/autoNameData.csv";
        String testFileLocation = "data/manualNameData.csv";
        char[] validCharacters = CharacterIterator.getDanishCharacterSet();
        return new CharacterIterator(fileLocation, testFileLocation, Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, validCharacters, epochSize, minimized);
    }

    public static CharacterIterator getEmbeddedIterator(int miniBatchSize, int sequenceLength, int epochSize,
                                                           boolean minimized) throws Exception {
        String fileLocation = "data/autoNameData.csv";
        String testFileLocation = "data/manualNameData.csv";
        char[] validCharacters = CharacterIterator.getDanishCharacterSet();
        return new EmbeddedCharacterIterator(fileLocation, testFileLocation, Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, validCharacters, epochSize, minimized);
    }

    public static CharacterIterator getSpeedIterator(int miniBatchSize, int sequenceLength, int epochSize,
                                                        boolean minimized) throws Exception {
        String fileLocation = "data/nameDataUnique.csv";
        char[] validCharacters = CharacterIterator.getDanishCharacterSet();
        return new ChSpeedIterator(fileLocation, Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, validCharacters, epochSize, minimized);
    }

    public static String[] convertTensorsToWords(INDArray output, CharacterIterator itr, int nCharactersToSample){
        int numOutputs = output.shape()[0];
        StringBuilder sb;
        String[] result = new String[numOutputs];
        for(int i=0; i < numOutputs; i++){
            sb = new StringBuilder(nCharactersToSample);
            for(int s = 0; s < output.shape()[2]; s++){     // for(char in outputWord)
                double[] outputProbDistr = new double[itr.totalOutcomes()];
                for(int j = 0; j < output.shape()[1]; j++)
                    outputProbDistr[j] = output.getDouble(i,j,s); // for(charProb in chararray)
                int wordIdx = getMax(outputProbDistr);      // TODO getMax might be better. I got no clue.
                if(wordIdx >= 0) sb.append(itr.convertIndexToCharacter(wordIdx));
                else sb.append("<NaN>");
            }
            result[i] = sb.toString();
        }
        return result;
    }

    public static int getMax(double[] distribution){
        int idx = 0;
        double max = 0;
        for(int i = 0; i < distribution.length; i++){
            if(distribution[i] > max){
                max = distribution[i];
                idx = i;
            }
        }
        if(max == 0) return -1;
        return idx;
    }

    public static char[] mergeArrays(char[] first, char[] second, char... extras){
        char[] mergedArray = new char[first.length +  second.length + extras.length];
        int i = 0;
        for(char c : first) mergedArray[i++] = c;
        for(char c : second) mergedArray[i++] = c;
        for(char c : extras) mergedArray[i++] = c;
        return mergedArray;

    }
}
