package neural_nets;


import neural_nets.anomalies.AnomaliesRNN;
import neural_nets.spellchecker.BiDirectionalRNN;
import java.io.IOException;
import java.util.Scanner;

public class DeepLearningTester {

    //Configuration variables
    private static String fileLocation = "data/dataAnomalies.csv";
    private static String testFileLocation = "data/dataAnomaliesTest.csv";
    private static String fileLocationRNN = "data/et_autoNameData.csv.merged";
    private static String testFileLocationRNN = "data/et_autoNameData.csv.test";
//    private static String fileLocationRNN = "data/shuffledTrainData.csv";
//    private static String testFileLocationRNN = "data/nameDataUnique.csv";
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
            double max = 0;
            double temp;
            int index = 0; // 670 atm!
            for(int i = 670; i < 675; i += 10){
                switch (type) {
                    case ANOMALY:
                        model = anomalyModel().loadModel(String.format("%sARNN_%s.bin", modelFilePathPrefix, i));
                        break;
                    case SPELLCHECKER:
                        model = spellCheckerModel().loadModel(String.format("%sBRNN_d3_et_2_%s.bin", modelFilePathPrefix, i));
                        break;
                }
                temp = model.runTesting(false);
                if(temp > max){
                    max = temp;
                    index = i;
                }
                System.out.println("Nr:" + i);
            }
        System.out.println(index);
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