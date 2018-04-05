package util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Alphabet {
    private static Alphabet alphabet = new Alphabet();
    private static final String dkAlphabet = "data/dk_alphabet.txt";
    public List<Character> alphabetChars = new ArrayList<>();
    private Alphabet(){
        try {
            BufferedReader reader = new BufferedReader(new FileReader(dkAlphabet));
            try {
                for (String s : reader.readLine().split(",,,")) alphabetChars.add(s.charAt(0));
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }
    public static Alphabet getInstance(){
        return alphabet;
    }
}