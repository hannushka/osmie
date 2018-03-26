package speed.neural_net;

import org.deeplearning4j.eval.Evaluation;
import util.Seq2Seq;

import java.io.File;

public class DLSpeedTester {
    public static void main(String[] args) {
        Seq2Seq model;
        java.io.File folder = new java.io.File("data/model_speed");
        java.io.File[] listOfFiles = folder.listFiles();
        String path = "";
        Evaluation maxEval = new Evaluation();
        int nbrOfClassesExcluded = Integer.MAX_VALUE;
        double maxF1Score = Double.MIN_VALUE;
        for (int i = 0 ; i < listOfFiles.length ; i++) {
            File file = listOfFiles[i];
            System.out.println("Processing file " + i + " / " + listOfFiles.length);
            try {
                model = SpeedNetwork.Builder().setCharacterIterator(SpeedNetwork.IteratorType.SPEED, false)
                        .loadModel(file.getPath());
                Evaluation eval = model.runTesting(false);
                if (nbrOfClassesExcluded > eval.averageF1NumClassesExcluded() ||
                        (eval.f1() > maxF1Score && nbrOfClassesExcluded >= eval.averageF1NumClassesExcluded())) {
                    path = file.getPath();
                    maxEval = eval;
                    maxF1Score = eval.f1();
                    nbrOfClassesExcluded = eval.averageF1NumClassesExcluded();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println(maxEval.stats());
        System.out.println(path);
    }
}
