package neural_nets.anomalies;

import neural_nets.Seq2Seq;

import java.util.Scanner;

public class PredictFromFile {
    private static String modelFilePathPrefix = "data/models/";
    private static String TEST_FILE = "data/anomaliesData.csv";

    public static void main(String[] args) {
        try {
            Scanner keyboard = new Scanner(System.in);
            String line;
            Seq2Seq model = AnomaliesRNN.Builder()
                    .setCharacterIterator("", TEST_FILE, Seq2Seq.IteratorType.ANOMALIES_PREDICT,
                            false, false)
                    .setNbrEpochs(1)
                    .loadModel(modelFilePathPrefix + "ARNN_490.bin");
            model.predict();
        } catch(Exception e) {
            System.out.println(e);
        }
    }
}