package neural_nets.anomalies;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class EncoderHelper {
    private static String[] maxSpeed = {"numeric-0", "numeric-30", "numeric-70", "missing", "other"};
    private static String[] ways = {"motorway", "primary", "residential", "pedestrian", "paths", "cycleway", "other"};
    private static String[] surfaces = {"paved", "paving_stones", "metal", "wood", "unpaved", "other"};

    private enum TagType {
        SPEED,
        SURFACE,
        HIGHWAY
    }

    public static String getSurfaceClass(String surface) {
        switch (surface) {
            case "paved":
            case "asphalt":
            case "concrete":
            case "concrete:lanes":
            case "concrete:plates":
                return surfaces[0];
            case "paving_stones":
            case "sett":
            case "unhewn_cobblestone":
            case "cobblestone":
                return surfaces[1];
            case "metal":
                return surfaces[2];
            case "wood":
                return surfaces[3];
            case "unpaved":
            case "compacted":
            case "fine_gravel":
            case "gravel":
            case "pebblestone":
            case "dirt":
            case "earth":
            case "grass":
            case "grass_paver":
            case "gravel_turf":
            case "ground":
            case "mud":
            case "sand":
            case "woodchips":
            case "salt":
            case "snow":
            case "ice":
                return surfaces[4];
            default:
                return surfaces[5];
        }
    }

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
        return getMapFromArray(TagType.SPEED, counter);
    }

    public static Map<String, Integer> getHighwayMap(int counter) {
        return getMapFromArray(TagType.HIGHWAY, counter);
    }

    public static Map<String, Integer> getSurfaceMap(int counter) {
        return getMapFromArray(TagType.SURFACE, counter);
    }

    public static Map<Character, Integer> getDanishCharacterSet(){
        try {
            Scanner scan = new Scanner(new File("data/wiki_alph.txt"));
            scan.useDelimiter(",,,");
            Map<Character, Integer> validChars = new HashMap<>();
            int counter = 0;
            while (scan.hasNext()) {
                String s = scan.next();
                if (s.length() > 0 && !validChars.containsKey(s.charAt(0))) {
                    validChars.put(s.charAt(0), counter++);
                }
            }
            validChars.put('\n', counter);
            // ^ Adding these here as they are not common to misspell
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

    private static Map<String, Integer> getMapFromArray(TagType type, int counter) {
        String[] array = null;
        switch (type) {
            case SPEED: array = maxSpeed;
                break;
            case SURFACE: array = surfaces;
                break;
            case HIGHWAY: array = ways;
                break;
        }
        Map<String, Integer> validChars = new HashMap<>();
        for (int i = 0 ; i < array.length ; i++) validChars.put(array[i], counter++);
        return validChars;
    }
}
