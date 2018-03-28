package util;

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

    /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
    private static char[] getMinimalCharacterSet(){
        List<Character> validChars = new LinkedList<>();
        for(char c='a'; c<='z'; c++) validChars.add(c);
        for(char c='A'; c<='Z'; c++) validChars.add(c);
        for(char c='0'; c<='9'; c++) validChars.add(c);
        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
        for(char c : temp) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i=0;
        for(Character c : validChars) out[i++] = c;
        return out;
    }

    public static char[] getDanishCharacterSet(){
        try (BufferedReader br = Files.newBufferedReader(Paths.get("data/dk_alphabet.txt"))) {
            Set<Character> validChars = new HashSet<>();
            char[] temp = {'!', '&', '-', '\'', '"', ',', '.', ' ', '\n', '\t', 'ü', 'ë', 'é'};

            for(char c : br.readLine().toLowerCase().toCharArray()) validChars.add(c);
            for(char c : temp) validChars.add(c);       // ^ Adding these here as they are not common to misspell
            char[] out = new char[validChars.size()];
            int i = 0;
            for(char c: validChars) out[i++] = c;

            return out;
        }catch (IOException ex){
            ex.printStackTrace();
        }
        return getMinimalCharacterSet();
    }
}
