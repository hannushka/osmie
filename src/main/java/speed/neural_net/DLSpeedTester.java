package speed.neural_net;

public class DLSpeedTester {
    public static void main(String[] args) {
        SpeedNetwork model;
        try {
            model = SpeedNetwork.Builder().setCharacterIterator(SpeedNetwork.IteratorType.TRANSLATER, 4000, true)
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
