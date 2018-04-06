package util;

import neural_nets.CharacterIterator;
import neural_nets.spellchecker.SpellCheckIterator;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Helper {
    public static String[] convertTensorsToWords(INDArray output, CharacterIterator itr){
        String[] result = new String[output.shape()[0]];
        for(int i=0; i < result.length; i++)
            result[i] = getWordFromDistr(getDoubleMatrixDistr(output.tensorAlongDimension(i, 1,2)), itr);
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
                     .collect(Collectors.joining())
                     .replaceAll("ü", "").trim();       // ü will be chosen if none is bigger supposedly.. TODO
    }

    public static int getIndexOfMax(double[] array){
        return IntStream.range(0, array.length)
                        .reduce((i, j)-> array[i] > array[j] ? i : j)
                        .getAsInt();
    }

    public static DeepSpellObject[] getSpellObjectsFromTensors(INDArray output, SpellCheckIterator itr){
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

    public static double getMaxDbl(double[] distribution){
        return Arrays.stream(distribution).max().getAsDouble();
    }

    public static void main(String[] args) {
//        System.out.println("hej".substring(0,0));
//        System.out.println("hej".substring(1));
    }
}
