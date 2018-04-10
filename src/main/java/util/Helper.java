package util;

import symspell.SymSpell;
import neural_nets.CharacterIterator;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Helper {
    public static String[] convertTensorsToWords(INDArray output, CharacterIterator itr){
        String[] result = new String[output.shape()[0]];
        for(int i=0; i < result.length; i++)
            result[i] = getWordFromDistr(getDoubleMatrixDistr(output.tensorAlongDimension(i, 1,2)), itr);
        return result;
    }

    public static double[][] getBestGuess(INDArray array){
        double[][] bestDistr = getDoubleMatrixDistr(array.tensorAlongDimension(0, 1,2)), tmpDistr;
        double maxScore = -1, tmpLowest = 10, tmpScore;
        for(int i=0; i < array.shape()[0]; i++){
            tmpDistr = getDoubleMatrixDistr(array.tensorAlongDimension(i, 1,2));
            for(double[] a : tmpDistr){
                tmpScore = getMaxDbl(a);
                if(tmpScore > 0) tmpLowest = Double.min(tmpLowest, tmpScore);
            }
            if(tmpLowest > maxScore){
                maxScore = tmpLowest;
                bestDistr = tmpDistr;
            }
        }
        return bestDistr;
    }

    private static double[][] getDoubleMatrixDistr(INDArray array){
        if(array.shape().length != 2) throw new DimensionMismatchException(array.shape().length, 2);
        NdIndexIterator iter = new NdIndexIterator(array.shape());
        double[][] distr = new double[array.shape()[1]][array.shape()[0]];
        iter.forEachRemaining(ints -> distr[ints[1]][ints[0]] = array.getDouble(ints[0],ints[1]));
        return distr;
    }

    public static String getWordFromDistr(double[][] wordDistr, CharacterIterator itr){
        return Arrays.stream(wordDistr)
                .map(Helper::getIndexOfMax)
                .map(itr::convertIndexToCharacter)
                .map(String::valueOf)
                .collect(Collectors.joining())      // TODO this method might be erronous now?
                .replaceAll("ü", "").trim();       // ü will be chosen if none is bigger supposedly.. TODO

    }

    public static int getIndexOfMax(double[] array){
        return IntStream.range(0, array.length)
                        .reduce((i, j)-> array[i] > array[j] ? i : j)
                        .getAsInt();
    }

    public static List<DeepSpellObject> getSpellObjectsFromUncertainTensors(INDArray input, INDArray output, INDArray labels,
                                                                            CharacterIterator itr){
        List<DeepSpellObject> objects = new ArrayList<>();
        double[][] wordMatrix, labelMatrix, inputMatrix;
        // inp: [a,b,c] -- tensorAlongDimension(i,1,2) returns tensors of shape [b,c].

        for(int i = 0; i < output.shape()[0]; i++){
            wordMatrix = getDoubleMatrixDistr(output.tensorAlongDimension(i,1,2));
            if(Arrays.stream(wordMatrix).anyMatch(array -> getMaxDbl(array) < 0.5 && getMaxDbl(array) > 0)){
                labelMatrix = getDoubleMatrixDistr(labels.tensorAlongDimension(i, 1,2));
                inputMatrix = getDoubleMatrixDistr(input.tensorAlongDimension(i, 1,2));
                DeepSpellObject deepSpellObject = new DeepSpellObject(wordMatrix, getWordFromDistr(wordMatrix, itr).trim(),
                        getWordFromDistr(labelMatrix, itr).trim(), getWordFromDistr(inputMatrix, itr).trim(), i);
                objects.add(deepSpellObject);
            }
        }
        return objects;
    }

    public static double getMaxDbl(double[] distribution){
        return Arrays.stream(distribution).max().getAsDouble();
    }

    public static void main(String[] args) throws FileNotFoundException {
        SymSpell symSpell = new SymSpell(-1, 2, -1, 10);
        if(!symSpell.loadDictionary("data/korpus_freq_dict.txt", 0, 1)) throw new FileNotFoundException();

        symSpell.lookupSpecialized("brokbjerghøs", SymSpell.Verbosity.Closest).forEach(System.out::println);

    }
}
