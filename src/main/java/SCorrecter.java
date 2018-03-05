import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SCorrecter {

    public static void run(SpellObject so) {
        List<String> commands = new ArrayList<>();
        commands.add("python3");
        commands.add("src/main/java/spellchecker/random_forest/s_correcter.py");
        commands.add(so.name.trim().toLowerCase());
        ProcessBuilder pb = new ProcessBuilder(commands);
        try {
            Process p = pb.start();
            List<String> outputList = IOUtils.readLines(p.getInputStream(), "utf-8");
            double score = Double.parseDouble(outputList.get(0));
            String correctedName = outputList.get(1);
            so.addCorrection(score, correctedName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
