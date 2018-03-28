package util;

import java.util.StringJoiner;

public class ArrayUtils {
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
}
