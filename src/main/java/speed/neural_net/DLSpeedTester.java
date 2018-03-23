package speed.neural_net;

import spellchecker.neural_net.Seq2Seq;

public class DLSpeedTester {
    public static void main(String[] args) {
        Seq2Seq model;
        try {
            model = SpeedNetwork.Builder().setCharacterIterator(SpeedNetwork.IteratorType.CLASSIC, true)
                    .loadModel(String.format("data/models/model%s.bin", 490));
//                    .loadModel("data/models/BiLSTM.bin");
            //            System.out.println(model.generateSuggestion("holstebrovej"));
//            model.runTestingOnTrain(true);
            model.runTesting(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
