package neural_nets;


import neural_nets.anomalies.AnomaliesRNN;

import java.util.Scanner;

public class DeepLearningTester {
    public static void main(String[] args) {
        String fileLocation = "data/dataAnomalies.csv";
        String testFileLocation = "data/dataAnomaliesTest.csv";
        String modelFilePathPrefix = "data/models/ARNN_";
        Seq2Seq model;
        try {
            Scanner keyboard = new Scanner(System.in);
            for(int i = 0; i < 1; i += 10){
                model = AnomaliesRNN.Builder()
                        .setCharacterIterator(fileLocation, testFileLocation, Seq2Seq.IteratorType.ANOMALIES, false)
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