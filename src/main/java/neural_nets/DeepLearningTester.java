package neural_nets;


import neural_nets.anomalies.AnomaliesRNN;
import neural_nets.spellchecker.BiDirectionalRNN;

import java.util.Scanner;

public class DeepLearningTester {
    public static void main(String[] args) {
        String fileLocation = "data/shuffledTrainData.csv";
        String testFileLocation = "data/manualNameData.csv";
        String modelFilePathPrefix = "data/models/BRNN_";
        Seq2Seq model;
        try {
            Scanner keyboard = new Scanner(System.in);
            for(int i = 10; i < 500; i += 10){
                model = BiDirectionalRNN.Builder()
                        .setCharacterIterator(fileLocation, testFileLocation, Seq2Seq.IteratorType.CLASSIC, false)
                        .loadModel(String.format("%s%s.bin", modelFilePathPrefix, i));
                model.runTesting(false);
                System.out.println("Nr:" + i);
                String a = keyboard.nextLine();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}