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
    private static String testFileLocationRNN = "data/manualNameData.csv";
    private static String modelFilePathPrefix = "data/models/";

    private static void testAnomalies() throws Exception {
        Seq2Seq model = AnomaliesRNN.Builder()
                        .setCharacterIterator(fileLocation, testFileLocation,
                                        Seq2Seq.IteratorType.ANOMALIES, false);
        runTest(model, String.format("%sARNN_", modelFilePathPrefix));
    }

    private static void testSpellChecker() throws Exception {
        Seq2Seq model = BiDirectionalRNN.Builder()
                        .setCharacterIterator(fileLocationRNN, testFileLocationRNN,
                                        Seq2Seq.IteratorType.CLASSIC, false);
        runTest(model, String.format("%sBRNN_", modelFilePathPrefix));

    }

    private static void runTest(Seq2Seq model, String path) throws IOException {
            Scanner keyboard = new Scanner(System.in);
            for(int i = 0; i < 500; i += 10){
                model.loadModel(String.format("%s%s.bin", path, i));
                model.runTesting(false);
                System.out.println("Nr:" + i);
                keyboard.nextLine();
            }
    }

    public static void main(String[] args) {
        try {
            testSpellChecker();
//            testAnomalies();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}