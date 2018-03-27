package spellchecker.neural_net;

import util.Seq2Seq;

import java.util.Scanner;

public class DeepLearningTester {
    public static void main(String[] args) {
        Seq2Seq model;
        try {
//            Scanner keyboard = new Scanner(System.in);
//            for(int i = 0; i < 500; i += 10){
                model = BiDirectionalRNN.Builder().setCharacterIterator(Seq2Seq.IteratorType.CLASSIC, false)
                        .loadModel(String.format("data/models/modelBRNN%s.bin", 480));
                model.runTesting(true);
//                System.out.println("Nr:" + i);
//                String a = keyboard.nextLine();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
