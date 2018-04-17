package neural_nets;

import neural_nets.spellchecker.BiDirectionalRNN;
import symspell.SymSpell;

import java.io.FileNotFoundException;
import java.util.Scanner;

public class NeuralNetDemo {
    private static String fileLocationRNN = "data/shuffledTrainData.csv";
    private static String testFileLocationRNN = "data/manualNameData.csv";
    private static String modelFilePathPrefix = "data/models/";

    private static void runDemo() throws Exception {
        Scanner keyboard = new Scanner(System.in);
        String line;
        SymSpell symSpell = new SymSpell(-1, 2, -1, 10);
        if(!symSpell.loadDictionary("data/korpus_freq_dict.txt", 0, 1)) try {
            throw new FileNotFoundException();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        Seq2Seq model = BiDirectionalRNN.Builder()
                    .setCharacterIterator(fileLocationRNN, testFileLocationRNN, Seq2Seq.IteratorType.CLASSIC, false)
                    .loadModel(modelFilePathPrefix + "BRNN_d4_best.bin");
//        model.runTesting(false);
        while(true){
            System.out.print("Enter input> ");
            line = keyboard.nextLine();
            model.printSuggestion(line, symSpell);
        }
    }

    public static void main(String[] args) {
        try {
            runDemo();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
