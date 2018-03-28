package util;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import speed.neural_net.ChSpeedIterator;
import spellchecker.neural_net.CharacterIterator;

import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.StringJoiner;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Helper {
    public static CharacterIterator getCharacterIterator(int miniBatchSize, int sequenceLength, int epochSize,
                                                         boolean minimized, boolean useCorpus) throws Exception {
        String fileLocation = "data/autoNameData.csv";
        String testFileLocation = "data/manualNameData.csv";
        char[] validCharacters = CharacterIterator.getDanishCharacterSet();
        return new CharacterIterator(fileLocation, testFileLocation, Charset.forName("UTF-8"),
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
            result[i] = getWordFromDistr(getDoubleMatrixDistr(output.tensorAlongDimension(i, 1,2)), itr)
                    .replaceAll("ü", "").trim();
            System.out.println(result[i]);
        }
        return result;
    }

    private static double[][] getDoubleMatrixDistr(INDArray array){
        if(array.shape().length != 2) throw new DimensionMismatchException(array.shape().length, 2);
        NdIndexIterator iter = new NdIndexIterator(array.shape());
        double[][] distr = new double[array.shape()[1]][array.shape()[0]];
        iter.forEachRemaining(ints -> distr[ints[1]][ints[0]] = array.getDouble(ints[0],ints[1]));
        return distr;
    }

    private static String getWordFromDistr(double[][] wordDistr, CharacterIterator itr){
        return Arrays.stream(wordDistr)
                     .map(Helper::getIndexOfMax)
                     .map(itr::convertIndexToCharacter)
                     .map(String::valueOf)
                     .collect(Collectors.joining());
    }

    private static int getIndexOfMax(double[] array){
        return IntStream.range(0, array.length)
                        .reduce((i, j)-> array[i] > array[j] ? i : j)
                        .getAsInt();
    }

    public static DeepSpellObject[] getSpellObjectsFromTensors(INDArray output, CharacterIterator itr){
        DeepSpellObject[] objects = new DeepSpellObject[output.shape()[0]];
        // inp: [a,b,c] -- tensorAlongDimension(i,1,2) returns tensors of shape [b,c].

        double[][] wordMatrix;
        for(int i = 0; i < objects.length; i++){
            wordMatrix = getDoubleMatrixDistr(output.tensorAlongDimension(i,1,2));
            DeepSpellObject deepSpellObject = new DeepSpellObject(wordMatrix, getWordFromDistr(wordMatrix, itr).trim());
            objects[i] = deepSpellObject;
        }
        return objects;
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

    public static char[] mergeArrays(String before, String after, char[]... arrays){
        StringJoiner joiner = new StringJoiner(" ");
        for(char[] array : arrays) joiner.add(String.valueOf(array));
        return (before + joiner.toString() + after).toCharArray();
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
        try {
            System.out.println(getWordFromDistr(new double[][]{new double[]{45,5,2,3,5,5,55,85,485}, new double[]{7,5,2,3}, new double[]{7,5,23,3}},
                    getCharacterIterator(32, 50, 10000, false, true)));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
