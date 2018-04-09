package util;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class EncoderHelper {

    private static String[] maxSpeed = {"numeric-0", "numeric-30", "numeric-70", "missing", "other"};
    private static String[] ways = {"motorway", "primary", "residential", "pedestrian", "paths", "cycleway", "other"};

    public static String getSpeedClass(String speed) {
        try {
            Integer numSpeed = Integer.parseInt(speed);
            if (numSpeed <= 30 && numSpeed >= 0) return maxSpeed[0];
            else if(numSpeed <= 70) return maxSpeed[1];
            else if (numSpeed > 70) return maxSpeed[2];
            else return maxSpeed[3]; //-1 or missing
        } catch(NumberFormatException e) {
            switch (speed) {
                case "walk":
                case "dk:rural":
                    return maxSpeed[4];
                default:
                    return maxSpeed[3];
            }
        }
    }

    public static String getHighwayClass(String highway) {
        switch (highway) {
            case "motorway":
            case "motorway_link":
            case "trunk":
            case "trunk_link":
                return ways[0];
            case "primary":
            case "primary_link":
            case "secondary":
            case "secondary_link":
            case "tertiary":
            case "tertiary_link":
            case "unclassified":
                return ways[1];
            case "residential":
                return ways[2];
            case "pedestrian":
            case "living_street":
                return ways[3];
            case "footway":
            case "bridleway":
            case "steps":
            case "path":
                return ways[4];
            case "cycleway":
                return ways[5];
            default:
                return ways[6];
        }
    }

    public static Map<String, Integer> getMaxSpeedMap(int counter) {
        Map<String, Integer> validChars = new HashMap<>();
        for (int i = 0 ; i < maxSpeed.length ; i++) validChars.put(maxSpeed[i], counter++);
        return validChars;
    }

    public static Map<String, Integer> getHighwayMap(int counter) {
        Map<String, Integer> validChars = new HashMap<>();
        for (int i = 0 ; i < ways.length ; i++) validChars.put(ways[i], counter++);
        return validChars;
    }

    public static Map<Character, Integer> getDanishCharacterSet(){
        try {
            Scanner scan = new Scanner(new File("data/dk_alphabet.txt"));
            scan.useDelimiter(",,,");
            Map<Character, Integer> validChars = new HashMap<>();
            char[] temp = {'!', '&', '-', '\'', '"', ',', '.', ' ', '\n', '\t', 'ü', 'ë', 'é'};
            int counter = 0;
            while (scan.hasNext()) {
                if (!validChars.containsKey(scan.next().charAt(0))) validChars.put(scan.next().charAt(0), counter++);
            }
            for(char c : temp) validChars.put(c, counter++);       // ^ Adding these here as they are not common to misspell
            scan.close();
            return validChars;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return getMinimalCharacterSet();
    }

    /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
    private static Map<Character, Integer> getMinimalCharacterSet(){
        Map<Character, Integer> validChars = new HashMap<>();
        int counter = 0;
        for(char c='a'; c<='z'; c++) validChars.put(c, counter++);
        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
        for(char c : temp) validChars.put(c, counter++);
        return validChars;
    }
}
