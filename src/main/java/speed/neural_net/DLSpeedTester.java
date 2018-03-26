package speed.neural_net;

import util.NgramBuilder;
import util.Seq2Seq;

public class DLSpeedTester {
    public static void main(String[] args) {
        Seq2Seq model;
        try {
            model = SpeedNetwork.Builder().setCharacterIterator(SpeedNetwork.IteratorType.SPEED, true)
                    .loadModel(String.format("data/model_speed/model%s.bin", 490));
//                    .loadModel("data/models/BiLSTM.bin");
        //            System.out.println(model.generateSuggestion("holstebrovej"));
//            model.runTestingOnTrain(true);
            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
//        NgramBuilder ngramBuilder = new NgramBuilder(3);
//        ngramBuilder.createNgramAlphabet("data/nameDataUnique.csv",
//                "data/ngramAlphabet.csv", 1);
//        ngramBuilder.loadNgramMap("data/ngramAlphabet.csv");
//        ngramBuilder.getFilteredNgrams("hemsøkneäåö").forEach(System.out::println);
    }
}
