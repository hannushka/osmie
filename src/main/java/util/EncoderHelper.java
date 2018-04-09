package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Scanner;

public class EncoderHelper {

    public static Map<String, Integer>  getMaxSpeedMap() throws IOException {
            Map<String, Integer> validChars = new HashMap<>();
            int counter = 0;
            String[] motorway = {"motorway", "motorway_link", "trunk", "trunk_link"};
            for (int i = 0 ; i < motorway.length ; i++) validChars.put(motorway[i], counter++);
            String[] primary = {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified"};
            for (int i = 0 ; i < primary.length ; i++) validChars.put(primary[i], counter++);
            validChars.put("residential", counter++);
            String[] pedestrian = {"pedestrian", "living_street"};
            for (int i = 0 ; i < pedestrian.length ; i++) validChars.put(pedestrian[i], counter++);
            String[] paths = {"footway", "bridleway", "steps", "path"};
            for (int i = 0 ; i < motorway.length ; i++) validChars.put(paths[i], counter++);
            validChars.put("cycleway", counter++);
            validChars.put("other", counter++);
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
