package util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Collectors;

public class NgramBuilder {
    private int nGram;
    private Set<String> nGrams;

    public NgramBuilder(int nGram){
        this.nGram = nGram;
    }

    private ArrayList<String> getNgrams(String word){
        ArrayList<String> wordNgrams = new ArrayList<>();
        for (int i = 0; i < word.length() - (nGram-1); i++) {
            wordNgrams.add(word.substring(i, i + (nGram)));
        }
        return wordNgrams;
    }

    public ArrayList<String> getFilteredNgrams(String word){
        ArrayList<String> wordNgrams = new ArrayList<>();
        String nGramSub;
        for (int i = 0; i < word.length() - (nGram-1); i++) {
            nGramSub = word.substring(i, i + (nGram));
            if(!nGrams.contains(nGramSub)) nGramSub = "!!!!!!!!!!!".substring(0, nGram);
            wordNgrams.add(nGramSub);
        }
        return wordNgrams;
    }

    public boolean loadNgramMap(String inputFilename){
        try {
            List<String> streets = Files.readAllLines(new File(inputFilename).toPath(), Charset.forName("UTF-8"));
            StringJoiner joiner = new StringJoiner(",,,");
            for(String s : streets) joiner.add(s);
            nGrams = Arrays.stream(joiner.toString().split(",,,")).collect(Collectors.toSet());
            return true;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }

    public boolean createNgramAlphabet(String inputFilename, String outputFilename, int filterSize){
        Map<String, Integer> nGramMap = new HashMap<>();
        try {
            List<String> streets = Files.readAllLines(new File(inputFilename).toPath(), Charset.forName("UTF-8"));
            for (String street : streets) {
                for (String nGram : getNgrams("\t" + street.toLowerCase() + "\n")) {
                    int reps = nGramMap.getOrDefault(nGram, 0);
                    nGramMap.put(nGram, reps + 1);
                }
            }
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outputFilename));
            StringJoiner joiner = new StringJoiner(",,,");
            for (Map.Entry<String, Integer> e : nGramMap.entrySet()) {
                if(e.getValue() > filterSize) joiner.add(e.getKey());
            }
            bufferedWriter.write(joiner.toString());
            bufferedWriter.close();
            return true;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }
}
