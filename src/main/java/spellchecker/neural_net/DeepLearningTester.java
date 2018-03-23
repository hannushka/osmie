package spellchecker.neural_net;

public class DeepLearningTester {
    public static void main(String[] args) {
        Seq2Seq model;
        try {
            model = RNN.Builder().setCharacterIterator(Seq2Seq.IteratorType.CLASSIC, false)
                    .loadModel(String.format("data/models/model%s.bin", 80 ));
//                    .loadModel("data/models/BiLSTM.bin");
            //            System.out.println(model.generateSuggestion("holstebrovej"));
//            model.runTestingOnTrain(true);
            model.runTesting(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
