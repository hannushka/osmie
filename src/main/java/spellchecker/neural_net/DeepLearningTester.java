package spellchecker.neural_net;

import util.Seq2Seq;

public class DeepLearningTester {
    public static void main(String[] args) {
        Seq2Seq model;
        try {
            model = BiDirectionalRNN.Builder().setCharacterIterator(Seq2Seq.IteratorType.CLASSIC, false)
                    .loadModel(String.format("data/models/modelBRNN%s.bin", 40));
//                    .loadModel("data/models/BiLSTM.bin");
            //            System.out.println(model.generateSuggestion("holstebrovej"));
//            model.runTestingOnTrain(true);
            model.runTesting(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
