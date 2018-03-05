import java.util.Map;
import java.util.TreeMap;

public class SpellObject {
    String name;
    Map<Double, Correction> corrections;

    public SpellObject(String name) {
        this.name = name;
        corrections = new TreeMap<>();
    }

    public void addCorrection(double score, String correctedName) {
        corrections.put(score, new Correction(score, correctedName));
    }

    public void print() {
        corrections.values().forEach(s -> System.out.println(s));
    }

    private class Correction {
        String correctedName;
        double score;

        public Correction(double score, String correctedName) {
            this.score = score;
            this.correctedName = correctedName;
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(correctedName);
            sb.append(" ");
            sb.append(score);
            return sb.toString();
        }
    }
}
