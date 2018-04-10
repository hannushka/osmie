package neural_nets;


import neural_nets.anomalies.AnomaliesRNN;
import neural_nets.spellchecker.BiDirectionalRNN;
import sun.awt.X11.AwtGraphicsConfigData;

import java.io.IOException;
import java.util.Scanner;

public class DeepLearningTester {

    //Configuration variables
    private static String fileLocation = "data/dataAnomalies.csv";
    private static String testFileLocation = "data/dataAnomaliesTest.csv";
    private static String modelFilePathPrefix = "data/models/";

    private enum ModelType {
        SPELLCHECKER,
        ANOMALY
    }

    public static Seq2Seq anomalyModel() throws Exception {
        return AnomaliesRNN.Builder()
                        .setCharacterIterator(fileLocation, testFileLocation, Seq2Seq.IteratorType.ANOMALIES, false);
    }

    public static Seq2Seq spellCheckerModel() throws Exception {
        return BiDirectionalRNN.Builder()
                .setCharacterIterator(fileLocation, testFileLocation, Seq2Seq.IteratorType.CLASSIC, false);

    }

    private static void runTest(ModelType type) throws Exception {
            Scanner keyboard = new Scanner(System.in);
            Seq2Seq model = null;
            for(int i = 0; i < 500; i += 10){
                switch (type) {
                    case ANOMALY:
                        model = anomalyModel().loadModel(String.format("%sARNN_%s.bin", modelFilePathPrefix, i));
                        break;
                    case SPELLCHECKER:
                        model = spellCheckerModel().loadModel(String.format("%sBRNN_%s.bin", modelFilePathPrefix, i));
                }
                model.runTesting(false);
                System.out.println("Nr:" + i);
                String a = keyboard.nextLine();
            }
    }

    public static void main(String[] args) {
        try {
            runTest(ModelType.ANOMALY);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}