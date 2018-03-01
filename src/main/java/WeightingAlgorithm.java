import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class WeightingAlgorithm {

    public static void runAlgorithm(String oldName, String newName) {
        ProcessBuilder pb = new ProcessBuilder("python3","src/main/java/spellchecker/random_forest/s_correcter.py");
        try {
            Process p = pb.start();
            BufferedReader output = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line;
            while ((line = output.readLine()) != null) {
                System.out.println(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
