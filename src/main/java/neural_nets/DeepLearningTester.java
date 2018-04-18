package neural_nets;


import neural_nets.anomalies.AnomaliesRNN;
import neural_nets.spellchecker.BiDirectionalRNN;
import java.io.IOException;
import java.util.Scanner;

public class DeepLearningTester {

    //Configuration variables
    private static String fileLocation = "data/dataAnomalies.csv";
    private static String testFileLocation = "data/dataAnomaliesTest.csv";
    private static String fileLocationRNN = "data/shuffledTrainData.csv";
    private static String testFileLocationRNN = "data/editDist1Data.csv";
    private static String modelFilePathPrefix = "data/models/";

    private enum ModelType {
        SPELLCHECKER,
        ANOMALY
    }

    public static Seq2Seq anomalyModel() throws Exception {
        return AnomaliesRNN.Builder()
                        .setCharacterIterator(fileLocation, testFileLocation, Seq2Seq.IteratorType.ANOMALIES,
                                false, false);
    }

    public static Seq2Seq spellCheckerModel() throws Exception {
        return BiDirectionalRNN.Builder()
                .setCharacterIterator(fileLocationRNN, testFileLocationRNN, Seq2Seq.IteratorType.CLASSIC,
                        false, false);

    }

    private static void runTest(ModelType type) throws Exception {
            Scanner keyboard = new Scanner(System.in);
            Seq2Seq model = null;
            for(int i = 10; i < 500; i += 10){
                switch (type) {
                    case ANOMALY:
                        model = anomalyModel().loadModel(String.format("%sARNN_%s.bin", modelFilePathPrefix, i));
                        break;
                    case SPELLCHECKER:
                        model = spellCheckerModel().loadModel("data/models/BRNN_d4_best.bin");
                        break;
                }
                model.runTesting(true);
                System.out.println("Nr:" + i);
                keyboard.nextLine();
            }
    }

    private static ModelType testType = ModelType.SPELLCHECKER;
    public static void main(String[] args) {
        try {
            runTest(testType);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}