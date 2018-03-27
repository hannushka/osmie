package util;

import org.nd4j.linalg.api.ndarray.INDArray;
import speed.neural_net.ChSpeedIterator;
import spellchecker.neural_net.CharacterIterator;
import spellchecker.neural_net.EmbeddedCharacterIterator;

import java.nio.charset.Charset;
import java.util.StringJoiner;

public class Helper {
    public static CharacterIterator getCharacterIterator(int miniBatchSize, int sequenceLength, int epochSize,
                                                         boolean minimized, boolean useCorpus) throws Exception {
        String fileLocation = "data/autoNameData.csv";
        String testFileLocation = "data/manualNameData.csv";
        char[] validCharacters = CharacterIterator.getDanishCharacterSet();
        return new CharacterIterator(fileLocation, testFileLocation, Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, validCharacters, epochSize, minimized, useCorpus);
    }

    public static CharacterIterator getEmbeddedIterator(int miniBatchSize, int sequenceLength, int epochSize,
                                                           boolean minimized, boolean useCorpus) throws Exception {
        String fileLocation = "data/autoNameData.csv";
        String testFileLocation = "data/manualNameData.csv";
        char[] validCharacters = CharacterIterator.getDanishCharacterSet();
        return new EmbeddedCharacterIterator(fileLocation, testFileLocation, Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, validCharacters, epochSize, minimized, useCorpus);
    }

    public static CharacterIterator getSpeedIterator(int miniBatchSize, int sequenceLength, int epochSize,
                                                        boolean minimized) throws Exception {
        String fileLocation = "data/speedData.csv";
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
            boolean found = false;
            for(int s = 0; s < output.shape()[2]; s++){     // for(char in outputWord)
                double[] outputProbDistr = new double[itr.totalOutcomes()];
                for(int j = 0; j < output.shape()[1]; j++)
                    outputProbDistr[j] = output.getDouble(i,j,s); // for(charProb in chararray)
                int wordIdx = getMax(outputProbDistr);      // TODO getMax might be better. I got no clue.
                if(wordIdx >= 0) sb.append(itr.convertIndexToCharacter(wordIdx));
                else sb.append("<NaN>");
                if(getMaxDbl(outputProbDistr) < 0.5 && wordIdx != -1){
                    found = true;
                    System.out.print(itr.convertIndexToCharacter(wordIdx)
                            + "("
                            + Double.toString(getMaxDbl(outputProbDistr)).substring(0,4)
                            + ","
                            + s
                            + ")"
                            + ",,,");
                }
            }
            result[i] = sb.toString();
            if(found) System.out.println(sb.toString().trim().replaceAll("<NaN>", ""));
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
    public static double getMaxDbl(double[] distribution){
        int idx = 0;
        double max = 0;
        for(int i = 0; i < distribution.length; i++){
            if(distribution[i] > max){
                max = distribution[i];
                idx = i;
            }
        }
        if(max == 0) return -1;
        return max;
    }

    public static char[] mergeArrays(char[]... arrays){
        StringJoiner joiner = new StringJoiner(" ");
        for(char[] array : arrays) joiner.add(String.valueOf(array));
        return ("\t" + joiner.toString() + "\n").toCharArray();
    }

    public static char[] mergeInArrays(char[] first, char[] second, char... extras){
        char[] mergedArray = new char[first.length +  second.length + extras.length + 2];
        int i = 1;
        for(char c : first) mergedArray[i++] = c;
        for(char c : extras) mergedArray[i++] = c;
        for(char c : second) mergedArray[i++] = c;
        mergedArray[0] = '\t';
        mergedArray[mergedArray.length-1] = '\n';
        return mergedArray;
    }

    public static char[] mergeOutArrays(char[] first, char[] second, char... extras){
        char[] mergedArray = new char[first.length +  second.length + extras.length + 2];
        int i = 1;
        for(char c : first) mergedArray[i++] = c;
        for(char c : extras) mergedArray[i++] = c;
        for(char c : second) mergedArray[i++] = c;
        mergedArray[0] = '\t';
        mergedArray[mergedArray.length-1] = '\n';
        return mergedArray;
    }

    public static Object[] mergeArrays(Object[]... arrays){
        int size = 0, i = 0;
        for(Object[] array : arrays) size += array.length;
        Object[] mergedArray = new Object[size];

        for(Object[] array : arrays){
            for(Object c : array) mergedArray[i++] = c;
        }
        return mergedArray;
    }

    public static String reduceEvalStats(String evalStats){
        String[] stats =evalStats.split("\n");
        StringBuilder statsSmall = new StringBuilder();
        for(int i = 0; i < 10; i++) statsSmall.append(stats[stats.length - (i+1)]).append("\n");
        return statsSmall.toString();
    }

    public static void main(String[] args) {
        for (char c : mergeArrays(new char[]{'h', 'e'}, new char[]{'e'})) {
            System.out.print(c);
        }
    }
}
