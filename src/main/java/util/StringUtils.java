package util;

import scala.Char;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class StringUtils {
    public static String reduceEvalStats(String evalStats){
        String[] stats = evalStats.split("\n");
        StringJoiner statsSmall = new StringJoiner("\n");
        for(int i = 0; i < 6; i++) statsSmall.add(stats[stats.length - (i+1)]);
        return statsSmall.toString();
    }

    public static boolean withinOneEditDist(String in, String label){
        EditDistance dist = new EditDistance(label);
        int distance = dist.DamerauLevenshteinDistance(in, 2);
        return distance <= 1 && distance >= 0;
    }

    public static boolean oneEditDist(String in, String label){
        EditDistance dist = new EditDistance(label);
        return dist.DamerauLevenshteinDistance(in, 2) == 1;
    }

    public static int editDist(String in, String label) {
        EditDistance dist = new EditDistance(label);
        return dist.DamerauLevenshteinDistance(in, 3);
    }
}
