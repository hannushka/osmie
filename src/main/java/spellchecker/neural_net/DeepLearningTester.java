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
                        .loadModel(String.format("data/models/modelBRNN_triple%s.bin", 170));
                model.runTesting(false);
//                System.out.println("Nr:" + i);
//                String a = keyboard.nextLine();
//            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
/**
 * String name
 * String label
 * Int position
 * double[] outputdistribution (position = probability for our choice)
 * When? Parameter-based!
 * Index for which str in list ish
 *
 * ----
 *
 * Metod: Skapa edit-dist1 ändringar
 * Replace baserat på outputdistr
 * Insert = allt?
 * Delete 1 case
 * Transpose 2 case
 * ^ Do this recursively, låt det propagera
 *
 * ----
 *
 * Test to remove if uncertain and just return inputclass. (Don't forget to test this)
 * Create stats for edit-dist = 1 also (?)
 * SymSpell. Testa att remove framifrån/bakifrån och enbart ändra char i fråga. OpenAddress check.

 **/